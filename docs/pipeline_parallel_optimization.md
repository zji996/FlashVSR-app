## FlashVSR 流水线并行与更激进的调度草案

本文档记录当前 FlashVSR 两卡流水线并行（`FLASHVSR_PP_*`）的实现要点，并给出一版**更激进的窗口级调度草案**，方便后续演进或对比实验。当前代码仍以已有实现为准，本文件只作为设计说明和调参参考。

---

## 1. 现有两卡调度回顾

相关代码：

- 环境变量解析：`backend/app/services/flashvsr_service.py` 中 `_parse_pipeline_parallel`。
- 流水线并行布置：`backend/app/flashvsr_core/diffsynth/pipelines/flashvsr_tiny_long.py:enable_pipeline_parallel`。
- 窗口级重叠调度：同文件 `__call__` 中 “Overlapped two-stage schedule (single-video acceleration)” 分支。

### 1.1 环境变量

- `FLASHVSR_PP_DEVICES="0,1"`  
  - 在 `enable_pipeline_parallel` 中被解析为 `["cuda:0", "cuda:1"]`。
  - `patch_embedding` 与前半段 blocks 放在 `cuda:0`；后半段 blocks 与 `head` 放在 `cuda:1`。
  - Cross-Attn 的持久 KV 缓存会迁移到各自 block 所在设备。
- `FLASHVSR_PP_SPLIT_BLOCK=auto`  
  - 切分点默认选在 `len(blocks)//2 - 1`，即按层数居中。
- `FLASHVSR_PP_OVERLAP=1`  
  - 在服务层调用 `pipe.enable_pipeline_overlap(True)`，使 `self.pp_overlap=True`。
  - 在 `__call__` 中进入“overlap 分支”，使用手写的 `_run_blocks_range(...)` 对 `[0..split]` 和 `[split+1..end]` 分段调度。

### 1.2 当前 overlap 时间线（简化）

对每个窗口 `t`，overlap 分支的大致执行顺序：

1. **前处理（主要在 GPU0）**
   - `LQ_proj_in.stream_forward` 逐块处理低清帧，构建本窗口的 `LQ_latents`。
   - 采样噪声 `cur_latents = _make_latents_window(t)`。
   - `patchify` 生成 `x_tokens`，构建 3D RoPE：`build_3d_freqs(...)` → `freqs_cpu` → `freqs_dev0/freqs_dev1`。
   - 将 `t_mod` 分别拷贝到 `dev0/dev1`。
2. **如存在上一窗口 `t-1` 的 pending：在 GPU1 上完成 Stage1**
   - `_run_blocks_range(split+1..end)`。
   - `head + unpatchify`，得到 `noise_pred_posi_prev`。
   - 在 GPU1 上更新上一窗口的 latent，并调用 `TCDecoder.decode_video` + 颜色校正。
3. **GPU0 上执行当前窗口的 Stage0**
   - `_run_blocks_range(0..split)`。
   - 使用 `torch.cuda.Stream(device=dev1)` 将 `x_mid` 异步拷贝到 GPU1（`x_mid_dev1`）。
4. **设置新的 pending**
   - 缓存当前窗口的中间结果和 LQ 索引，用于下一轮迭代完成 Stage1+decode。

循环结束后，对最后一个 pending 窗口再做一次 Stage1+decode。

**KV cache 约束：**

- 对于同一 block，其时间维度上的 `pre_cache_k/pre_cache_v` 必须按窗口顺序累积，因此无法并行计算窗口 `t` 和 `t+1` 的**同一 block**。
- 当前 overlap 通过“block 维度切分”来解决这个问题：
  - Stage0 只读写 `block[0..split]` 的 KV；
  - Stage1 只读写 `block[split+1..end]` 的 KV；
  - 因此可以在 Stage1(t)（GPU1）运行时，在 GPU0 上执行 Stage0(t+1)。

**结果：**  
即使开启 `FLASHVSR_PP_OVERLAP=1`，窗口内部仍然存在较长的“只用 GPU0”（前处理）和“只用 GPU1”（decode + 部分 Stage1）的阶段，`nvidia-smi` 上通常会看到两张卡呈交替高负载，而不是始终 100% 平滑打满。

---

## 2. 更激进的窗口流水线草案（3-stage pipeline）

目标：在不改变数学结果（与单卡保持数值等价）的前提下，尽量让两张卡在绝大多数时间内都有工作可做，把目前的“2-stage（Stage0/Stage1）重叠”提升为**3-stage** 窗口流水线：

1. **Stage A（GPU0 为主）**：LQ 预处理 + latent 采样 + patchify + RoPE 预计算  
2. **Stage B（GPU0）**：DiT 前半段 blocks（Stage0）  
3. **Stage C（GPU1）**：DiT 后半段 blocks（Stage1）+ `head` + `TCDecoder.decode_video` + 颜色校正

在时间维度上，为每个窗口 `t` 维护一个状态：

- `A_t`：窗口 t 的前处理结果（`LQ_latents_t`、`cur_latents_t`、`x_tokens_t`、`freqs_cpu_t` 等）。
- `B_t`：窗口 t 在 split 处的中间激活 `x_mid_t`（已经在 GPU1 上有一份拷贝）。
- `C_t`：窗口 t 的最终帧片段（decode 完成，已搬回 CPU）。

理想的 steady-state 时间线（t ≥ 2）：

- GPU0：同时负责 `B_t` 和 `A_{t+1}`（前一窗口的 Stage0 + 下一窗口的前处理）。
- GPU1：执行 `C_{t-1}`（上一窗口的 Stage1 + decode）。

### 2.1 状态机草案

用伪结构描述每个窗口的状态：

```python
class WindowCtx(NamedTuple):
    index: int
    # Stage A 输出
    cur_latents: torch.Tensor          # 噪声 latent（仍在 dev0）
    LQ_latents: Optional[list[Tensor]]
    f: int; h: int; w: int
    freqs_cpu: torch.Tensor            # RoPE base（CPU）
    LQ_pre_idx: int
    LQ_cur_idx: int

    # Stage B 输出（可延后填充）
    x_mid_dev1: Optional[torch.Tensor] = None
```

高层调度伪代码（省略错误处理与边界情况）：

```python
windows: Deque[WindowCtx] = deque()

# 1) 预先算出 t=0 的 Stage A
windows.append( stage_A_prepare(0, prev_ctx=None) )

for t in range(process_total_num):
    cur = windows[-1]           # 当前要执行 Stage0 的窗口
    prev = windows[-2] if len(windows) >= 2 else None

    # (C) 先在 GPU1 上完成 t-1 的 Stage1 + decode（如果存在）
    if prev is not None and prev.x_mid_dev1 is not None:
        run_stage1_and_decode(prev)   # 使用独立 stream + dev1

    # (B) 在 GPU0 上执行当前窗口的 Stage0，并异步把 x_mid 复制到 GPU1
    cur = cur._replace(x_mid_dev1=run_stage0_and_copy(cur))
    windows[-1] = cur

    # (A) 在 GPU0 上为下一个窗口 t+1 准备前处理（LQ_proj + patchify + RoPE）
    if t + 1 < process_total_num:
        next_ctx = stage_A_prepare(t + 1, prev_ctx=cur)
        windows.append(next_ctx)

# 循环结束后，再补最后一个窗口的 Stage1 + decode
last = windows[-1]
run_stage1_and_decode(last)
```

关键点：

- `stage_A_prepare` 尽量只依赖 `LQ_pre_idx/LQ_cur_idx` 和 `cur_process_idx`，不会写入任何 KV cache，因此可以与前一窗口的 Stage1/Stage0 并行。
- Stage0/Stage1 仍按 block ID 对 KV cache 分段，保证顺序一致性。
- `LQ_pre_idx/LQ_cur_idx` 的推进与 `_release_lq_frames` 的调用依然保持现有窗口逻辑，只是调用时机稍作调整（更多地与 Stage1/decode 重叠）。

### 2.2 与现有实现的差异

相对当前 overlap 版本，主要变化有：

1. **明确拆出 Stage A**  
   - 现有实现中，LQ_proj + latent 采样 + patchify + RoPE 生成与 Stage0/Stage1 串在一个循环里，导致每个窗口开头有一段“只有 GPU0” 的时间。
   - 新草案将这些操作打包为 Stage A，并尝试在 GPU1 处理上一窗口时就提前执行。
2. **窗口队列 + 状态对象**  
   - 替代当前的 `pending` 单一变量，用 `deque` 管理多个 in-flight 窗口，使得“t 的 Stage0 + t+1 的 StageA + t-1 的 Stage1”在 steady-state 更自然。
3. **更清晰的设备职责划分**  
   - GPU0：`LQ_proj_in`、噪声采样、patchify、Stage0。  
   - GPU1：Stage1、head、TCDecoder、颜色校正。  
   - RoPE 和 `t_mod` 的 CPU→GPU 拷贝统一放在 Stage A 内，避免在 Stage0/Stage1 中重复判断。

数值上，只要：

- 各窗口对 LQ 帧的索引 `(start_idx, end_idx)` 保持不变；
- `cur_process_idx` → `build_3d_freqs` 的 `f_offset` 未更改；
- KV cache 的写入顺序仍严格是「窗口从小到大，block 从小到大」；

则重排执行顺序不会改变最终输出，只影响 GPU 的占用形态。

---

## 3. 实现建议与渐进式推进

考虑到改动涉及长视频流式推理的关键路径，推荐按以下顺序推进，而不是一次性大改：

1. **阶段一：文档 + profiling**
   - 保持现有代码不变，仅通过 `torch.cuda.Event` 或 `nvtx.range_push/pop` 在 overlap 分支中打标，观察每个阶段（LQ_proj、Stage0、Stage1、decode）的耗时与 GPU 利用率。
   - 根据结果估计更合理的 `pp_split_idx`，必要时在 `.env` 中手动设置具体层号而非 `auto`。
2. **阶段二：轻量预取**
   - 在现有 overlap 循环中，先触发上一窗口的 Stage1 + decode，再执行当前窗口的 LQ_proj / patchify，这样可以让 GPU0 的前处理更大程度地与 GPU1 的 Stage1/decode 重叠。
   - 这一阶段只需要调整几行调用顺序，风险较小。
3. **阶段三：引入 `WindowCtx` + Stage A 抽象（可受控启用）**
   - 新增一个受控开关，例如：
     - `FLASHVSR_PP_OVERLAP_MODE=basic|aggressive`（默认 `basic` 为当前实现）。
   - 当为 `aggressive` 时，启用 `WindowCtx` 状态机和 3-stage pipeline；否则走原有 overlap 路径。
   - 这样即使实验失败，用户也可以通过环境变量快速回退。

代码层面的修改建议（示意，不代表已经实现）：

- 在 `FlashVSRTinyLongPipeline.__call__` 中：
  - 将现有 overlap 分支重构为：
    - `if use_overlap and overlap_mode == "aggressive":` → 调用新的 `_run_overlapped_pipeline_aggressive(...)`；
    - `elif use_overlap:` → 保留当前实现（basic）。
  - 新增 `_run_overlapped_pipeline_aggressive` 函数，内部实现本文档描述的 3-stage 状态机。

---

## 4. 调参与观测建议

在有两张类似规格的 GPU（如 2×3080 或 2×4090）时，可以按如下顺序探索：

1. **基础设置**
  - `FLASHVSR_PP_DEVICES="0,1"`
  - `FLASHVSR_PP_SPLIT_BLOCK=auto`（或根据 profiling 手动调整）。
  - `FLASHVSR_PP_OVERLAP=1`
2. **测量基线**
  - 先在 `FLASHVSR_PP_OVERLAP=0` 下跑一次长视频，记录总耗时与两张卡的平均/峰值占用。
   - 再开启 overlap（basic），比较吞吐和输出一致性。
   - 每个任务现在会记录 `started_at` 与 `finished_at`（见 `/api/tasks/{id}` 返回值），可以直接用 `finished_at - started_at` 或 `video_info.processing_time` 统计不同模式下的端到端耗时。
3. **尝试 aggressive 模式（实现后）**
   - 打开 `FLASHVSR_PP_OVERLAP_MODE=aggressive`（或类似开关）。
   - 对比：
     - 总耗时是否进一步下降；
     - GPU0/GPU1 利用率是否更平滑；
     - 输出视频是否与 basic 模式 bit-wise 近似（允许浮点级别差异）。

---

## 5. 后续工作

本草案的目标是明确“能做什么”以及“做了之后如何验证”。推荐的后续步骤：

1. 在 overlap 分支中先加上轻量 profiling（阶段一+二），确保对现有路径的性能瓶颈有量化认知。
2. 在此基础上再评估是否值得引入 `WindowCtx` + `overlap_mode=aggressive` 的实现复杂度。
3. 若 aggressive 模式实际收益有限（例如受制于磁盘 I/O 或 CPU 解码），可以仅保留本文档作为研究记录，无需在生产代码中启用。

如有新的实验结果（比如不同 split、分辨率、显存策略的对比），建议附加到本文件下方，方便后续查阅与复现。

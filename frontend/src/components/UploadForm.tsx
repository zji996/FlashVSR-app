/**
 * 视频上传表单组件
 */

import { useEffect, useMemo, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { isAxiosError } from 'axios';
import { tasksApi } from '../api/tasks';
import { systemApi } from '../api/system';
import { ModelVariant, type TaskParameters } from '../types';
import Snackbar from './Snackbar';

const PREPROCESS_WIDTH_OPTIONS = [640, 768, 896, 960, 1024, 1152, 1280];
const PRESET_PROFILES = [
  {
    key: '1080p',
    label: '接近 1080p',
    description: '预处理 960px + 2× 超分，适合高清流媒体素材',
    preprocess_width: 960,
    scale: 2.0,
  },
  {
    key: '2k',
    label: '锐利 2K',
    description: '预处理 1152px + 2×，在 16:9 视频上接近 2304px',
    preprocess_width: 1152,
    scale: 2.0,
  },
  {
    key: 'fast',
    label: '快速出图',
    description: '预处理 768px + 2×，更省显存的批量模式',
    preprocess_width: 768,
    scale: 2.0,
  },
];
const SUPPORTED_EXTENSIONS = [
  '.mp4',
  '.mov',
  '.avi',
  '.mkv',
  '.ts',
  '.m2ts',
  '.mts',
  '.m4s',
  '.mpg',
  '.mpeg',
  '.webm',
];
const SUPPORTED_LABEL = SUPPORTED_EXTENSIONS.map((ext) => ext.replace('.', '').toUpperCase()).join(', ');

export default function UploadForm() {
  const queryClient = useQueryClient();
  const { data: systemStatus } = useQuery({
    queryKey: ['system-status'],
    queryFn: systemApi.getStatus,
    staleTime: 10000,
  });
  const [file, setFile] = useState<File | null>(null);
  const [parameters, setParameters] = useState<TaskParameters>({
    scale: 2.0,
    sparse_ratio: 2.0,
    local_range: 11,
    seed: 0,
    model_variant: ModelVariant.TINY_LONG,
    preprocess_width: 640,
  });
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [clientError, setClientError] = useState<string | null>(null);
  const [snackbar, setSnackbar] = useState<{ message: string; variant: 'success' | 'error' } | null>(null);

  useEffect(() => {
    if (!snackbar) {
      return;
    }
    const timer = window.setTimeout(() => setSnackbar(null), 4000);
    return () => window.clearTimeout(timer);
  }, [snackbar]);

  const showSnackbar = (message: string, variant: 'success' | 'error') => {
    setSnackbar({ message, variant });
  };

  const readyVariants = systemStatus?.flashvsr?.ready_variants ?? {};
  const tinyLongReady = readyVariants?.[ModelVariant.TINY_LONG];
  const preprocessWidthSelectValue = PREPROCESS_WIDTH_OPTIONS.includes(parameters.preprocess_width)
    ? String(parameters.preprocess_width)
    : 'custom';

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'video/*': SUPPORTED_EXTENSIONS,
    },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        setFile(acceptedFiles[0]);
      }
    },
  });

  const uploadMutation = useMutation({
    mutationFn: (data: { file: File; parameters: TaskParameters }) =>
      tasksApi.createTask(data.file, data.parameters),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
      setFile(null);
      setClientError(null);
      showSnackbar('任务创建成功！', 'success');
    },
    onError: (error: unknown) => {
      const message = (() => {
        if (isAxiosError<{ detail?: string | string[] }>(error)) {
          const detail = error.response?.data?.detail;
          if (typeof detail === 'string') {
            return detail;
          }
          if (Array.isArray(detail)) {
            return detail.join(', ');
          }
          return error.message;
        }
        if (error instanceof Error) {
          return error.message;
        }
        return '未知错误';
      })();

      showSnackbar(`上传失败: ${message}`, 'error');
      setClientError(message);
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setClientError(null);
    if (!file) {
      setClientError('请选择视频文件');
      return;
    }

    // Validate preprocess width
    if (!parameters.preprocess_width || parameters.preprocess_width < 128) {
      setClientError('预处理宽度必须不小于 128 像素');
      return;
    }
    if (!tinyLongReady) {
      setClientError('模型权重尚未就绪，无法创建任务');
      return;
    }
    uploadMutation.mutate({ file, parameters });
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  const approxOutputWidth = useMemo(() => {
    if (!parameters.preprocess_width || !parameters.scale) return null;
    const scaled = parameters.preprocess_width * parameters.scale;
    if (Number.isNaN(scaled) || scaled <= 0) return null;
    const aligned = Math.floor(scaled / 128) * 128;
    return aligned > 0 ? aligned : null;
  }, [parameters.preprocess_width, parameters.scale]);

  const isPresetActive = (preset: (typeof PRESET_PROFILES)[number]) =>
    parameters.preprocess_width === preset.preprocess_width && parameters.scale === preset.scale;

  const disableSubmit = !file || uploadMutation.isPending || tinyLongReady === false;

  const handlePresetClick = (preset: (typeof PRESET_PROFILES)[number]) => {
    setParameters({
      ...parameters,
      preprocess_width: preset.preprocess_width,
      scale: preset.scale,
    });
    setShowAdvanced(false);
  };

  const clearFile = () => {
    setFile(null);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6 max-w-4xl mx-auto w-full px-4 sm:px-0">
      <div className="card">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-2xl font-bold">上传视频</h2>
            <p className="text-sm text-gray-500 mt-1">
              选择素材 → 设定预处理宽度/超分倍数 → 一键提交，前端会实时显示任务进度。
            </p>
          </div>
          {systemStatus?.flashvsr && (
            <div className="rounded-xl bg-gray-50 px-4 py-3 text-sm">
              <div className="font-medium text-gray-800">FlashVSR {systemStatus.flashvsr.version}</div>
              <div className="text-gray-500">
                Tiny Long 状态：
                <span className={`font-semibold ${tinyLongReady ? 'text-green-600' : 'text-red-600'}`}>
                  {tinyLongReady ? '可用' : '缺少权重'}
                </span>
              </div>
            </div>
          )}
        </div>
        {clientError && (
          <div className="mt-4 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {clientError}
          </div>
        )}
      </div>

      {/* 文件上传区域 */}
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-8 sm:p-10 lg:p-12 text-center cursor-pointer
          transition-colors
          ${isDragActive ? 'border-primary-500 bg-primary-50' : 'border-gray-300 hover:border-primary-400'}
          ${file ? 'bg-green-50 border-green-500' : ''}
        `}
      >
        <input {...getInputProps()} />
        {file ? (
          <div className="space-y-2">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <div className="text-lg font-medium text-green-700">✓ 已选择文件</div>
                <div className="text-gray-700 break-all">{file.name}</div>
                <div className="text-sm text-gray-500 mt-1">{formatFileSize(file.size)}</div>
              </div>
              <button
                type="button"
                onClick={clearFile}
                className="text-sm text-red-500 hover:text-red-600 underline"
              >
                重新选择
              </button>
            </div>
            <p className="text-xs text-gray-500">
              支持格式：{SUPPORTED_LABEL}，更少见的容器会自动转码为 MP4。
            </p>
          </div>
        ) : isDragActive ? (
          <div className="text-lg text-primary-600">
            放开以上传视频...
          </div>
        ) : (
          <div>
            <div className="text-lg text-gray-700 mb-2">
              拖拽视频文件到此处，或点击选择文件
            </div>
            <div className="text-sm text-gray-500">
              支持 {SUPPORTED_LABEL} 等格式，其它视频也会自动转码为 MP4。
            </div>
          </div>
        )}
      </div>

      {/* 参数配置 */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="card lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">预处理宽度</h3>
            <span className="text-xs text-gray-500">必选项</span>
          </div>
          <div className="space-y-3">
            <select
              value={preprocessWidthSelectValue}
              onChange={(e) => {
                if (e.target.value === 'custom') {
                  setParameters({
                    ...parameters,
                    preprocess_width: parameters.preprocess_width,
                  });
                  return;
                }
                setParameters({
                  ...parameters,
                  preprocess_width: parseInt(e.target.value, 10),
                });
              }}
              className="input"
            >
              {PREPROCESS_WIDTH_OPTIONS.map((width) => (
                <option key={width} value={width}>
                  {width} px
                </option>
              ))}
              <option value="custom">自定义</option>
            </select>
            {preprocessWidthSelectValue === 'custom' && (
              <input
                type="number"
                min="640"
                step="128"
                value={parameters.preprocess_width}
                onChange={(e) =>
                  setParameters({
                    ...parameters,
                    preprocess_width: e.target.value ? parseInt(e.target.value, 10) : 640,
                  })
                }
                className="input"
              />
            )}
          </div>
          <p className="text-xs text-gray-500 mt-1">
            始终预处理：请选择常用档位，或自定义宽度（建议 640-1280）。常见值如 960 配合 2× 超分可接近 1080p。
          </p>
        </div>
        <div className="card space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">快捷预设</h3>
            <button
              type="button"
              className="text-xs text-primary-600 hover:text-primary-700"
              onClick={() =>
                setParameters({
                  ...parameters,
                  preprocess_width: 640,
                  scale: 2.0,
                })
              }
            >
              重置为默认
            </button>
          </div>
          <div className="space-y-3">
            {PRESET_PROFILES.map((preset) => {
              const active = isPresetActive(preset);
              return (
                <button
                  type="button"
                  key={preset.key}
                  onClick={() => handlePresetClick(preset)}
                  className={`w-full rounded-lg border px-4 py-3 text-left transition ${
                    active ? 'border-primary-500 bg-primary-50 shadow-sm' : 'border-gray-200 hover:border-primary-300'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-gray-800">{preset.label}</span>
                    {active && <span className="text-xs text-primary-600">当前</span>}
                  </div>
                  <p className="text-sm text-gray-500 mt-1">{preset.description}</p>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      <div className="card space-y-3">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h3 className="text-lg font-semibold">模型 & 输出</h3>
            <p className="text-sm text-gray-500">
              FlashVSR v1.1 Tiny Long（固定变体）。根据预处理宽度和倍数估算输出尺寸。
            </p>
          </div>
          <div className="rounded-full bg-gray-100 px-4 py-1 text-sm text-gray-700">
            预计输出宽度 ≈ {approxOutputWidth ? `${approxOutputWidth}px` : '—'}（高度随原视频比例对齐）
          </div>
        </div>
        {tinyLongReady === false && systemStatus?.flashvsr && (
          <p className="text-xs text-red-600">
            缺少权重: {systemStatus.flashvsr.missing_files.join(', ') || '请参考 README 下载。'}
          </p>
        )}
        <div className="rounded-lg border border-dashed border-gray-200 px-4 py-3 text-sm text-gray-600">
          Tiny Long 针对长序列/逐帧图片优化。后端会自动把最终输入对齐到 128 的倍数，以满足 WanVideo 的窗口需求。
        </div>
      </div>

      <div className="card">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">高级参数（可选）</h3>
          <button
            type="button"
            onClick={() => setShowAdvanced((prev) => !prev)}
            className="text-xs text-primary-600 hover:text-primary-700 flex items-center gap-1"
          >
            <span>{showAdvanced ? '折叠' : '展开'}</span>
          </button>
        </div>
        {showAdvanced && (
          <div className="mt-4 grid grid-cols-1 gap-6 md:grid-cols-2">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">超分倍数 (Scale)</label>
              <input
                type="number"
                min="1"
                max="8"
                step="0.1"
                value={parameters.scale}
                onChange={(e) => setParameters({ ...parameters, scale: parseFloat(e.target.value) })}
                className="input"
              />
              <p className="text-xs text-gray-500 mt-1">推荐值: 2.0</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">稀疏比率 (Sparse Ratio)</label>
              <input
                type="number"
                min="1"
                max="4"
                step="0.1"
                value={parameters.sparse_ratio}
                onChange={(e) =>
                  setParameters({
                    ...parameters,
                    sparse_ratio: parseFloat(e.target.value),
                  })
                }
                className="input"
              />
              <p className="text-xs text-gray-500 mt-1">推荐值: 1.5 (快) 或 2.0 (稳定)</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">局部范围 (Local Range)</label>
              <input
                type="number"
                min="7"
                max="15"
                step="2"
                value={parameters.local_range}
                onChange={(e) =>
                  setParameters({
                    ...parameters,
                    local_range: parseInt(e.target.value),
                  })
                }
                className="input"
              />
              <p className="text-xs text-gray-500 mt-1">推荐值: 9 (更锐利) 或 11 (更稳定)</p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">随机种子 (Seed)</label>
              <input
                type="number"
                min="0"
                value={parameters.seed}
                onChange={(e) => setParameters({ ...parameters, seed: parseInt(e.target.value) })}
                className="input"
              />
              <p className="text-xs text-gray-500 mt-1">0 为随机</p>
            </div>
          </div>
        )}
      </div>

      {/* 提交按钮 */}
      <div className="card">
        <div className="flex flex-col gap-3">
          <div className="text-sm text-gray-500">
            系统会把视频输出到 <code>storage/results</code> 并自动合并音频。长视频默认启用分片导出，即使任务失败也会保留已完成片段。
          </div>
          <button
            type="submit"
            disabled={disableSubmit}
            className={`btn btn-primary w-full text-lg py-3 ${disableSubmit ? 'opacity-60 cursor-not-allowed' : ''}`}
          >
            {uploadMutation.isPending ? '上传中...' : '开始处理'}
          </button>
        </div>
      </div>

      {snackbar && (
        <Snackbar
          message={snackbar.message}
          variant={snackbar.variant}
          onClose={() => setSnackbar(null)}
        />
      )}
    </form>
  );
}

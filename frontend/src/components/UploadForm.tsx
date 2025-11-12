/**
 * 视频上传表单组件
 */

import { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { isAxiosError } from 'axios';
import { tasksApi } from '../api/tasks';
import { systemApi } from '../api/system';
import { ModelVariant, type TaskParameters } from '../types';

const PREPROCESS_OPTIONS: Array<{
  value: TaskParameters['preprocess_strategy'];
  label: string;
  description: string;
}> = [
  { value: 'none', label: '关闭', description: '直接送入 FlashVSR，不做额外采样。' },
  {
    value: 'always',
    label: '开启',
    description: '无条件在 GPU 推理前把素材缩放到指定宽度，并按原比例计算高度。',
  },
];

const PREPROCESS_WIDTH_OPTIONS = [640, 768, 896, 1024, 1152];
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
    scale: 4.0,
    sparse_ratio: 2.0,
    local_range: 11,
    seed: 0,
    model_variant: ModelVariant.TINY_LONG,
    preprocess_strategy: 'none',
    preprocess_width: 640,
  });

  const variantOptions: Array<{ value: ModelVariant; label: string; description: string }> = [
    {
      value: ModelVariant.TINY_LONG,
      label: 'Tiny Long（默认）',
      description: '长序列/逐帧图片友好版本，针对 8n-3 帧裁切优化。',
    },
    {
      value: ModelVariant.TINY,
      label: 'Tiny',
      description: '4× 推理，显存占用最低，适合绝大多数视频。',
    },
    {
      value: ModelVariant.FULL,
      label: 'Full（最高画质）',
      description: '全尺寸模型，需要 Wan2.1 VAE 权重与更高显存。',
    },
  ];

  const readyVariants = systemStatus?.flashvsr?.ready_variants ?? {};
  const selectedVariant = variantOptions.find((option) => option.value === parameters.model_variant);
  const selectedVariantReady = readyVariants?.[parameters.model_variant];
  const preprocessWidthSelectValue =
    parameters.preprocess_width && PREPROCESS_WIDTH_OPTIONS.includes(parameters.preprocess_width)
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
      alert('任务创建成功！');
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

      alert(`上传失败: ${message}`);
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      alert('请选择视频文件');
      return;
    }
    uploadMutation.mutate({ file, parameters });
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <form onSubmit={handleSubmit} className="card max-w-3xl mx-auto">
      <h2 className="text-2xl font-bold mb-6">上传视频</h2>

      {/* 文件上传区域 */}
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-12 text-center cursor-pointer
          transition-colors
          ${isDragActive ? 'border-primary-500 bg-primary-50' : 'border-gray-300 hover:border-primary-400'}
          ${file ? 'bg-green-50 border-green-500' : ''}
        `}
      >
        <input {...getInputProps()} />
        {file ? (
          <div>
            <div className="text-lg font-medium text-green-700 mb-2">
              ✓ 已选择文件
            </div>
            <div className="text-gray-700">{file.name}</div>
            <div className="text-sm text-gray-500 mt-1">
              {formatFileSize(file.size)}
            </div>
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
      <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="md:col-span-2">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            模型变体
          </label>
          <select
            value={parameters.model_variant}
            onChange={(e) =>
              setParameters({
                ...parameters,
                model_variant: e.target.value as ModelVariant,
              })
            }
            className="input"
          >
            {variantOptions.map((option) => {
              const ready = readyVariants?.[option.value] ?? true;
              return (
                <option key={option.value} value={option.value} disabled={!ready}>
                  {option.label}
                  {!ready ? '（权重未就绪）' : ''}
                </option>
              );
            })}
          </select>
          <p className="text-xs text-gray-500 mt-1">
            {selectedVariant?.description ?? '选择不同变体以平衡显存与画质需求。'}
            {systemStatus?.flashvsr?.version
              ? ` · FlashVSR ${systemStatus.flashvsr.version}`
              : ''}
          </p>
          {selectedVariantReady === false && systemStatus?.flashvsr && (
            <p className="text-xs text-red-600 mt-1">
              缺少权重: {systemStatus.flashvsr.missing_files.join(', ') || '请参考 README 下载。'}
            </p>
          )}
        </div>
        <div className="md:col-span-2">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            预处理策略
          </label>
          <select
            value={parameters.preprocess_strategy}
            onChange={(e) =>
              setParameters({
                ...parameters,
                preprocess_strategy: e.target.value as TaskParameters['preprocess_strategy'],
              })
            }
            className="input"
          >
            {PREPROCESS_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          <p className="text-xs text-gray-500 mt-1">
            {PREPROCESS_OPTIONS.find((opt) => opt.value === parameters.preprocess_strategy)?.description}
          </p>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            预处理宽度 (128 的倍数)
          </label>
          <div className="space-y-3">
            <select
              value={preprocessWidthSelectValue}
              onChange={(e) => {
                if (e.target.value === 'custom') {
                  setParameters({
                    ...parameters,
                    preprocess_width:
                      parameters.preprocess_width &&
                      !PREPROCESS_WIDTH_OPTIONS.includes(parameters.preprocess_width)
                        ? parameters.preprocess_width
                        : null,
                  });
                  return;
                }
                setParameters({
                  ...parameters,
                  preprocess_width: parseInt(e.target.value, 10),
                });
              }}
              disabled={parameters.preprocess_strategy === 'none'}
              className="input"
            >
              {PREPROCESS_WIDTH_OPTIONS.map((width) => (
                <option key={width} value={width}>
                  {width} px
                </option>
              ))}
              <option value="custom">自定义</option>
            </select>
            {preprocessWidthSelectValue === 'custom' && parameters.preprocess_strategy !== 'none' && (
              <input
                type="number"
                min="128"
                step="128"
                value={parameters.preprocess_width ?? ''}
                onChange={(e) =>
                  setParameters({
                    ...parameters,
                    preprocess_width: e.target.value ? parseInt(e.target.value, 10) : null,
                  })
                }
                className="input"
              />
            )}
          </div>
          <p className="text-xs text-gray-500 mt-1">
            {parameters.preprocess_strategy === 'none'
              ? '关闭预处理时无需填写'
              : '从常用档位中选择，或使用自定义输入满足特殊素材。'}
          </p>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            超分倍数 (Scale)
          </label>
          <input
            type="number"
            min="1"
            max="8"
            step="0.1"
            value={parameters.scale}
            onChange={(e) =>
              setParameters({ ...parameters, scale: parseFloat(e.target.value) })
            }
            className="input"
          />
          <p className="text-xs text-gray-500 mt-1">推荐值: 2.0</p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            稀疏比率 (Sparse Ratio)
          </label>
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
          <label className="block text-sm font-medium text-gray-700 mb-2">
            局部范围 (Local Range)
          </label>
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
          <p className="text-xs text-gray-500 mt-1">
            推荐值: 9 (更锐利) 或 11 (更稳定)
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            随机种子 (Seed)
          </label>
          <input
            type="number"
            min="0"
            value={parameters.seed}
            onChange={(e) =>
              setParameters({ ...parameters, seed: parseInt(e.target.value) })
            }
            className="input"
          />
          <p className="text-xs text-gray-500 mt-1">0 为随机</p>
        </div>
      </div>

      {/* 提交按钮 */}
      <div className="mt-8">
        <button
          type="submit"
          disabled={!file || uploadMutation.isPending}
          className="btn btn-primary w-full text-lg py-3"
        >
          {uploadMutation.isPending ? '上传中...' : '开始处理'}
        </button>
      </div>
    </form>
  );
}

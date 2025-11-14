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

const PREPROCESS_WIDTH_OPTIONS = [640, 768, 896, 960, 1024, 1152, 1280];
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

    // Validate preprocess width
    if (!parameters.preprocess_width || parameters.preprocess_width < 128) {
      alert('预处理宽度必须不小于 128 像素');
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
            模型
          </label>
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
            <div>
              <div className="text-sm font-medium text-gray-900">
                Tiny Long（固定）
              </div>
              <p className="text-xs text-gray-500">
                专为长序列和逐帧图片优化的 FlashVSR v1.1 Tiny Long 变体。
              </p>
            </div>
            {systemStatus?.flashvsr && (
              <div className="flex items-center gap-2">
                <span
                  className={`inline-flex items-center rounded-full px-2 py-1 text-xs font-medium ${
                    tinyLongReady
                      ? 'bg-green-50 text-green-700 ring-1 ring-green-600/20'
                      : 'bg-red-50 text-red-700 ring-1 ring-red-600/20'
                  }`}
                >
                  {tinyLongReady ? '权重已就绪' : '权重缺失'}
                </span>
                <span className="text-xs text-gray-500">
                  FlashVSR {systemStatus.flashvsr.version}
                </span>
              </div>
            )}
          </div>
          {tinyLongReady === false && systemStatus?.flashvsr && (
            <p className="text-xs text-red-600 mt-1">
              缺少权重: {systemStatus.flashvsr.missing_files.join(', ') || '请参考 README 下载。'}
            </p>
          )}
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            预处理宽度 (像素)
          </label>
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
        <div className="md:col-span-2">
          <button
            type="button"
            onClick={() => setShowAdvanced((prev) => !prev)}
            className="text-xs text-primary-600 hover:text-primary-700 flex items-center gap-1"
          >
            <span>{showAdvanced ? '隐藏高级参数' : '显示高级参数（可选）'}</span>
          </button>
        </div>

        {showAdvanced && (
          <>
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
          </>
        )}
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

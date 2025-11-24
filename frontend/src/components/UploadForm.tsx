/**
 * 视频上传表单组件
 */

import { useEffect, useMemo, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { isAxiosError } from 'axios';
import { tasksApi } from '../api/tasks';
import { systemApi } from '../api/system';
import {
  ModelVariant,
  type TaskParameters,
  type TaskParameterFieldMeta,
  type TaskPresetProfileMeta,
} from '../types';
import Snackbar from './Snackbar';

const FALLBACK_PREPROCESS_WIDTH_OPTIONS = [640, 768, 896, 960, 1024, 1152, 1280];
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
  const { data: parameterSchema } = useQuery({
    queryKey: ['task-parameter-schema'],
    queryFn: tasksApi.getParameterSchema,
    staleTime: Infinity,
  });
  const [file, setFile] = useState<File | null>(null);
  const [parameters, setParameters] = useState<TaskParameters>({
    scale: 2.0,
    sparse_ratio: 2.0,
    local_range: 11,
    seed: 0,
    model_variant: ModelVariant.TINY_LONG,
    preprocess_width: 640,
    preserve_aspect_ratio: false,
  });
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [clientError, setClientError] = useState<string | null>(null);
  const [snackbar, setSnackbar] = useState<{ message: string; variant: 'success' | 'error' } | null>(null);
  const [useCustomWidth, setUseCustomWidth] = useState(false);

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

  const preprocessField = parameterSchema?.fields.find(
    (field) => field.name === 'preprocess_width'
  );
  const preprocessWidthOptions = useMemo(() => {
    const values =
      preprocessField?.recommended
        ?.map((opt) => Number(opt.value))
        .filter((v) => Number.isFinite(v) && v > 0) ?? [];
    if (values.length === 0) {
      return FALLBACK_PREPROCESS_WIDTH_OPTIONS;
    }
    return Array.from(new Set(values)).sort((a, b) => a - b);
  }, [preprocessField]);

  const presetProfiles: TaskPresetProfileMeta[] = useMemo(
    () => parameterSchema?.presets ?? [],
    [parameterSchema]
  );

  const preprocessWidthSelectValue = useCustomWidth
    ? 'custom'
    : preprocessWidthOptions.includes(parameters.preprocess_width)
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

  const isPresetActive = (preset: TaskPresetProfileMeta) =>
    parameters.preprocess_width === preset.preprocess_width &&
    parameters.scale === preset.scale;

  const disableSubmit = !file || uploadMutation.isPending || tinyLongReady === false;

  const handlePresetClick = (preset: TaskPresetProfileMeta) => {
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
    <form onSubmit={handleSubmit} className="space-y-6 w-full">
      {/* 错误提示 */}
      {clientError && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {clientError}
        </div>
      )}

      {/* 文件上传区域 */}
      <div
        {...getRootProps()}
        className={`
          border-4 border-dashed rounded-2xl p-8 sm:p-10 text-center cursor-pointer
          transition-all duration-200 flex flex-col justify-center min-h-[240px]
          shadow-lg hover:shadow-2xl
          ${isDragActive ? 'border-primary-500 bg-gradient-to-br from-primary-50 to-indigo-50 scale-[1.03] ring-4 ring-primary-200' : 'border-primary-400 bg-gradient-to-br from-blue-50 to-indigo-50 hover:border-primary-500 hover:scale-[1.01]'}
          ${file ? 'bg-gradient-to-br from-green-50 to-emerald-50 border-green-500 ring-4 ring-green-200' : ''}
        `}
      >
          <input {...getInputProps()} />
          {file ? (
            <div className="space-y-3">
              <div className="flex flex-col gap-3">
                <div>
                  <div className="text-2xl font-bold text-green-700 mb-2">✓ 已选择文件</div>
                  <div className="text-lg font-medium text-gray-800 break-all">{file.name}</div>
                  <div className="text-sm text-gray-600 mt-2">{formatFileSize(file.size)}</div>
                </div>
                <button
                  type="button"
                  onClick={clearFile}
                  className="text-sm text-red-600 hover:text-red-700 font-medium underline mt-2"
                >
                  重新选择
                </button>
              </div>
              <p className="text-xs text-gray-600 mt-4">
                支持格式：{SUPPORTED_LABEL}，更少见的容器会自动转码为 MP4。
              </p>
            </div>
          ) : isDragActive ? (
            <div>
              <div className="text-3xl mb-3">📹</div>
              <div className="text-2xl font-bold text-primary-600">
                放开以上传视频...
              </div>
            </div>
          ) : (
            <div>
              <div className="text-5xl mb-4">📤</div>
              <div className="text-xl font-bold text-gray-800 mb-3">
                拖拽视频文件到此处，或点击选择文件
              </div>
              <div className="text-sm text-gray-600">
                支持 {SUPPORTED_LABEL} 等格式，其它视频也会自动转码为 MP4。
              </div>
            </div>
          )}
      </div>

      {/* 参数配置 - 统一的完整配置区 */}
      <div className="card bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 border-2 border-blue-200 shadow-lg">
        <div className="grid gap-6 lg:grid-cols-2">
          {/* 左侧：预处理宽度 */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xl font-bold text-gray-900">预处理宽度</h3>
              <span className="text-xs font-semibold text-primary-700 bg-primary-100 px-3 py-1 rounded-full border border-primary-200">必选项</span>
            </div>
            <div className="space-y-2">
              <select
                value={preprocessWidthSelectValue}
                onChange={(e) => {
                  if (e.target.value === 'custom') {
                    setUseCustomWidth(true);
                    return;
                  }
                  setUseCustomWidth(false);
                  setParameters({
                    ...parameters,
                    preprocess_width: parseInt(e.target.value, 10),
                  });
                }}
                className="input"
              >
                {preprocessWidthOptions.map((width) => (
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
            <p className="text-sm text-gray-600">
              请选择常用档位，或输入自定义宽度（建议 640-1280，例如 960 搭配 2× 超分接近 1080p）。
            </p>
            
            {/* 预计输出宽度 */}
            <div className="mt-4 pt-4 border-t-2 border-blue-200">
              <div className="rounded-xl bg-white border-2 border-primary-200 px-5 py-4 text-center shadow-md">
                <div className="font-semibold text-gray-700 text-xs uppercase tracking-wide mb-1">预计输出宽度</div>
                <div className="text-3xl font-bold text-primary-600">
                  {approxOutputWidth ? `${approxOutputWidth}px` : '—'}
                </div>
              </div>
            </div>
          </div>

          {/* 右侧：快捷预设 */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xl font-bold text-gray-900">快捷预设</h3>
              <button
                type="button"
                className="text-sm text-primary-600 hover:text-primary-700 font-semibold underline"
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
            <div className="grid grid-cols-1 gap-3">
              {presetProfiles.map((preset) => {
                const active = isPresetActive(preset);
                return (
                  <button
                    type="button"
                    key={preset.key}
                    onClick={() => handlePresetClick(preset)}
                    className={`w-full rounded-xl border-2 px-4 py-3.5 text-left transition-all shadow-sm hover:shadow-md ${
                      active 
                        ? 'border-primary-500 bg-white ring-2 ring-primary-300 shadow-lg' 
                        : 'border-gray-300 bg-white hover:border-primary-400'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-bold text-gray-900 truncate">{preset.label}</span>
                      {active && <span className="ml-2 text-xs text-white bg-primary-600 px-2 py-0.5 rounded-full font-semibold">✓</span>}
                    </div>
                    <p className="text-xs leading-relaxed text-gray-600">{preset.description}</p>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* 高级参数 - 整合在同一卡片内 */}
        <div className="mt-6 pt-6 border-t-2 border-blue-200">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold text-gray-900">高级参数（可选）</h3>
            <button
              type="button"
              onClick={() => setShowAdvanced((prev) => !prev)}
              className="text-sm text-primary-600 hover:text-primary-700 font-semibold flex items-center gap-1.5 px-3 py-1.5 rounded-lg hover:bg-white/50 transition-colors border border-primary-200"
            >
              <span>{showAdvanced ? '▲ 折叠' : '▼ 展开'}</span>
            </button>
          </div>
          {showAdvanced && (
            <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
              {parameterSchema?.fields
                .filter((field) => field.ui_group === 'advanced')
                .map((field: TaskParameterFieldMeta) => {
                  const key = field.name as keyof TaskParameters;
                  const value = parameters[key] as number | boolean | undefined;

                  if (field.field_type === 'boolean') {
                    const checked = Boolean(value);
                    return (
                      <div
                        key={field.name}
                        className="sm:col-span-2 lg:col-span-4 flex items-center gap-2 mt-2"
                      >
                        <input
                          id={field.name}
                          type="checkbox"
                          checked={checked}
                          onChange={(e) =>
                            setParameters({
                              ...parameters,
                              [key]: e.target.checked,
                            })
                          }
                          className="h-4 w-4 rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                        />
                        <div className="flex flex-col gap-1">
                          <label htmlFor={field.name} className="text-sm text-gray-700">
                            {field.label}
                          </label>
                          {field.description && (
                            <p className="text-xs text-gray-500">{field.description}</p>
                          )}
                        </div>
                      </div>
                    );
                  }

                  const min = field.min ?? undefined;
                  const max = field.max ?? undefined;
                  const step = field.step ?? undefined;

                  return (
                    <div key={field.name}>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        {field.label}
                      </label>
                      <input
                        type="number"
                        min={min}
                        max={max ?? undefined}
                        step={step}
                        value={value ?? ''}
                        onChange={(e) =>
                          setParameters({
                            ...parameters,
                            [key]: e.target.value === '' ? value : Number(e.target.value),
                          })
                        }
                        className="input"
                      />
                      {field.recommended.length > 0 ? (
                        <p className="text-xs text-gray-500 mt-1">
                          推荐值:{' '}
                          {field.recommended
                            .map((opt) => opt.description || opt.label)
                            .join(' / ')}
                        </p>
                      ) : (
                        field.description && (
                          <p className="text-xs text-gray-500 mt-1">{field.description}</p>
                        )
                      )}
                    </div>
                  );
                })}
            </div>
          )}
        </div>
      </div>

      {/* 权重警告 */}
      {tinyLongReady === false && systemStatus?.flashvsr && (
        <div className="rounded-xl border-2 border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700">
          <span className="font-bold">⚠ 缺少权重:</span> {systemStatus.flashvsr.missing_files.join(', ') || '请参考 README 下载。'}
        </div>
      )}

      {/* 提交按钮 */}
      <div className="card border-2 border-green-200 bg-gradient-to-br from-green-50 to-emerald-50">
        <div className="flex flex-col gap-4">
          <div className="text-sm text-gray-700 leading-relaxed bg-white rounded-lg px-4 py-3 border border-gray-200">
            <span className="font-semibold text-gray-900">📁 输出说明：</span>系统会把视频输出到 <code className="bg-gray-100 text-gray-700 px-2 py-0.5 rounded font-mono text-xs">storage/results</code> 并自动合并音频。长视频默认启用分片导出，即使任务失败也会保留已完成片段。
          </div>
          <button
            type="submit"
            disabled={disableSubmit}
            className={`w-full text-xl py-5 font-bold rounded-xl shadow-xl transition-all duration-200 ${
              disableSubmit 
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed' 
                : 'bg-gradient-to-r from-primary-600 to-indigo-600 text-white hover:from-primary-700 hover:to-indigo-700 hover:shadow-2xl hover:scale-[1.02]'
            }`}
          >
            {uploadMutation.isPending ? '⏳ 上传中...' : '🚀 开始处理'}
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

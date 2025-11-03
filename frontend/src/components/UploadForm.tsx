/**
 * 视频上传表单组件
 */

import { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { tasksApi } from '../api/tasks';
import type { TaskParameters } from '../types';

export default function UploadForm() {
  const queryClient = useQueryClient();
  const [file, setFile] = useState<File | null>(null);
  const [parameters, setParameters] = useState<TaskParameters>({
    scale: 4.0,
    sparse_ratio: 2.0,
    local_range: 11,
    seed: 0,
  });

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'video/*': ['.mp4', '.mov', '.avi', '.mkv'],
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
    onError: (error: any) => {
      alert(`上传失败: ${error.response?.data?.detail || error.message}`);
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
              支持 MP4, MOV, AVI, MKV 格式
            </div>
          </div>
        )}
      </div>

      {/* 参数配置 */}
      <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
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
          <p className="text-xs text-gray-500 mt-1">推荐值: 4.0</p>
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


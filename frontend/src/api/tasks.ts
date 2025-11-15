/**
 * 任务相关API
 */

import apiClient from './client';
import type { Task, TaskListResponse, TaskProgressResponse, TaskParameters } from '../types';

export const tasksApi = {
  /**
   * 创建任务并上传视频
   */
  async createTask(file: File, parameters: TaskParameters): Promise<Task> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('scale', parameters.scale.toString());
    formData.append('sparse_ratio', parameters.sparse_ratio.toString());
    formData.append('local_range', parameters.local_range.toString());
    formData.append('seed', parameters.seed.toString());
    // 模型变体在后端固定为 tiny_long，这里不再发送字段，保留在 TaskParameters 仅用于响应展示。
    formData.append('preprocess_width', parameters.preprocess_width.toString());

    const response = await apiClient.post<Task>('/api/tasks/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  /**
   * 获取任务列表
   */
  async getTasks(
    page: number = 1,
    pageSize: number = 20,
    status?: string
  ): Promise<TaskListResponse> {
    const params: Record<string, number | string> = { page, page_size: pageSize };
    if (status) {
      params.status = status;
    }

    const response = await apiClient.get<TaskListResponse>('/api/tasks/', { params });
    return response.data;
  },

  /**
   * 获取任务详情
   */
  async getTask(taskId: string): Promise<Task> {
    const response = await apiClient.get<Task>(`/api/tasks/${taskId}`);
    return response.data;
  },

  /**
   * 获取任务进度
   */
  async getTaskProgress(taskId: string): Promise<TaskProgressResponse> {
    const response = await apiClient.get<TaskProgressResponse>(
      `/api/tasks/${taskId}/progress`
    );
    return response.data;
  },

  /**
   * 基于磁盘分片导出当前可恢复的部分结果
   */
  async exportFromChunks(taskId: string): Promise<TaskProgressResponse> {
    const response = await apiClient.post<TaskProgressResponse>(
      `/api/tasks/${taskId}/export_from_chunks`
    );
    return response.data;
  },

  /**
   * 删除任务
   */
  async deleteTask(taskId: string): Promise<void> {
    await apiClient.delete(`/api/tasks/${taskId}`);
  },

  /**
   * 获取结果文件URL
   */
  getResultUrl(taskId: string): string {
    return `${apiClient.defaults.baseURL}/api/files/${taskId}/result`;
  },

  /**
   * 获取输入文件URL
   */
  getInputUrl(taskId: string): string {
    return `${apiClient.defaults.baseURL}/api/files/${taskId}/input`;
  },

  /**
   * 获取结果预览图URL
   */
  getPreviewUrl(taskId: string): string {
    return `${apiClient.defaults.baseURL}/api/files/${taskId}/preview`;
  },
};

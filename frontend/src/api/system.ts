/**
 * 系统信息API
 */

import apiClient from './client';
import type { SystemStatus } from '../types';

export const systemApi = {
  /**
   * 获取系统状态
   */
  async getStatus(): Promise<SystemStatus> {
    const response = await apiClient.get<SystemStatus>('/api/system/status');
    return response.data;
  },
};


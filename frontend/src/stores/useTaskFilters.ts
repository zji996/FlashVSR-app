import { create } from 'zustand';
import type { TaskStatus } from '../types';

interface TaskFilterState {
  page: number;
  pageSize: number;
  statusFilter: TaskStatus | '';
  setPage: (page: number) => void;
  setPageSize: (pageSize: number) => void;
  setStatusFilter: (status: TaskStatus | '') => void;
  reset: () => void;
}

export const useTaskFilterStore = create<TaskFilterState>((set) => ({
  page: 1,
  pageSize: 8,
  statusFilter: '',
  setPage: (page) => set({ page }),
  setPageSize: (pageSize) => set({ pageSize, page: 1 }),
  setStatusFilter: (statusFilter) => set({ statusFilter, page: 1 }),
  reset: () => set({ page: 1, pageSize: 8, statusFilter: '' }),
}));

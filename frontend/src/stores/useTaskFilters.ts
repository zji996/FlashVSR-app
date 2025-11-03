import { create } from 'zustand';
import { TaskStatus } from '../types';

interface TaskFilterState {
  page: number;
  statusFilter: TaskStatus | '';
  setPage: (page: number) => void;
  setStatusFilter: (status: TaskStatus | '') => void;
  reset: () => void;
}

export const useTaskFilterStore = create<TaskFilterState>((set) => ({
  page: 1,
  statusFilter: '',
  setPage: (page) => set({ page }),
  setStatusFilter: (statusFilter) => set({ statusFilter, page: 1 }),
  reset: () => set({ page: 1, statusFilter: '' }),
}));

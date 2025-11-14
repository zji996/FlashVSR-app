/**
 * 类型定义
 */

export const TaskStatus = {
  PENDING: "pending",
  PROCESSING: "processing",
  COMPLETED: "completed",
  FAILED: "failed",
} as const;

export type TaskStatus = (typeof TaskStatus)[keyof typeof TaskStatus];

export const ModelVariant = {
  TINY_LONG: 'tiny_long',
} as const;

export type ModelVariant = (typeof ModelVariant)[keyof typeof ModelVariant];

export const MODEL_VARIANT_LABELS: Record<ModelVariant, string> = {
  [ModelVariant.TINY_LONG]: 'Tiny Long',
};

export interface TaskParameters {
  scale: number;
  sparse_ratio: number;
  local_range: number;
  seed: number;
  model_variant: ModelVariant;
  preprocess_width: number;
}

export interface VideoInfo {
  width?: number;
  height?: number;
  total_frames?: number;
  fps?: number;
  duration?: number;
  processed_frames?: number;
  processing_time?: number;
  inference_time?: number;
  bit_rate?: number;
  avg_frame_rate?: number;
  preprocess_applied?: boolean;
  preprocess_width?: number;
  preprocess_result_width?: number;
  preprocess_result_height?: number;
  predicted_output_width?: number;
  predicted_output_height?: number;
}

export interface Task {
  id: string;
  created_at: string;
  updated_at: string;
  status: TaskStatus;
  input_file_name: string;
  output_file_name?: string;
  video_info?: VideoInfo;
  parameters: TaskParameters;
  progress: number;
  total_frames?: number;
  processed_frames: number;
  estimated_time_remaining?: number;
  error_message?: string;
}

export interface TaskListResponse {
  tasks: Task[];
  total: number;
  page: number;
  page_size: number;
}

export interface TaskProgressResponse {
  task_id: string;
  status: TaskStatus;
  progress: number;
  processed_frames: number;
  total_frames?: number;
  estimated_time_remaining?: number;
  error_message?: string;
}

export interface SystemStatus {
  gpu_available: boolean;
  gpu_info?: {
    name: string;
    count: number;
    memory_allocated: number;
    memory_reserved: number;
    memory_total: number;
  };
  tasks: {
    total: number;
    pending: number;
    processing: number;
    completed: number;
    failed: number;
  };
  flashvsr?: {
    version: string;
    default_variant: ModelVariant;
    available_variants: ModelVariant[];
    ready_variants: Partial<Record<ModelVariant, boolean>>;
    missing_files: string[];
    model_path?: string;
    weights_ready: boolean;
  };
}

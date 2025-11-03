/**
 * 进度条组件
 */

interface ProgressBarProps {
  progress: number;
  processedFrames: number;
  totalFrames?: number;
}

export default function ProgressBar({
  progress,
  processedFrames,
  totalFrames,
}: ProgressBarProps) {
  return (
    <div>
      <div className="flex justify-between text-sm text-gray-600 mb-2">
        <span>进度: {progress.toFixed(1)}%</span>
        {totalFrames && (
          <span>
            {processedFrames} / {totalFrames} 帧
          </span>
        )}
      </div>
      <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
        <div
          className="bg-primary-600 h-full transition-all duration-300 rounded-full"
          style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
        />
      </div>
    </div>
  );
}


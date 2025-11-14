import type { ReactNode } from 'react';

interface SnackbarProps {
  message: ReactNode;
  variant?: 'success' | 'error' | 'info';
  onClose?: () => void;
}

const variantStyles: Record<Required<SnackbarProps>['variant'], string> = {
  success: 'bg-emerald-600 text-white',
  error: 'bg-red-600 text-white',
  info: 'bg-gray-900 text-white',
};

export default function Snackbar({ message, variant = 'info', onClose }: SnackbarProps) {
  return (
    <div className="fixed inset-x-4 bottom-6 z-50 sm:inset-x-auto sm:left-auto sm:right-6 sm:min-w-[280px]">
      <div
        className={`flex items-center justify-between gap-4 rounded-2xl px-4 py-3 text-sm shadow-lg ring-1 ring-black/10 ${variantStyles[variant]}`}
      >
        <div className="flex-1 leading-relaxed">{message}</div>
        {onClose && (
          <button
            type="button"
            onClick={onClose}
            className="text-xs font-semibold uppercase tracking-wide opacity-80 hover:opacity-100"
          >
            关闭
          </button>
        )}
      </div>
    </div>
  );
}


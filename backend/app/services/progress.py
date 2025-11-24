from __future__ import annotations

import time
from typing import Callable


class ProgressReporter:
    """
    Thin wrapper around a frame-based progress callback.

    The core export / job code only talks to this class and remains
    agnostic of Celery or database details.
    """

    def __init__(
        self,
        total_frames: int,
        on_update: Callable[[int, int, float], None],
    ) -> None:
        self.total_frames = max(int(total_frames), 0)
        self.processed_frames = 0
        self._on_update = on_update
        self._start = time.time()

    def report(self, processed_delta: int) -> None:
        """Report additional processed frames."""
        if processed_delta < 0:
            processed_delta = 0
        self.processed_frames += processed_delta
        elapsed = max(time.time() - self._start, 0.0)
        avg = elapsed / self.processed_frames if self.processed_frames else 0.0
        self._on_update(self.processed_frames, self.total_frames, avg)

    def done(self) -> None:
        """Mark progress as finished."""
        if self.total_frames:
            # 保证最终一次回调总是使用 total_frames。
            self._on_update(self.total_frames, self.total_frames, 0.0)


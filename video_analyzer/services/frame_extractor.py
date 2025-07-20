"""
FrameExtractor class for extracting frames from videos.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Iterator
import logging
import math

import cv2
import numpy as np

from video_analyzer.config.frame_extractor import (
    FrameExtractorConfig,
    ExtractionStrategy,
)
from video_analyzer.models.video import Frame, VideoData
from video_analyzer.utils.errors import VideoProcessingError


logger = logging.getLogger(__name__)


class FrameExtractor:
    """
    Extracts frames from videos using different strategies.
    """

    def __init__(self, config: Optional[FrameExtractorConfig] = None):
        """
        Initialize the FrameExtractor with the given configuration.

        Args:
            config: Configuration for the FrameExtractor. If None, default configuration is used.
        """
        self.config = config or FrameExtractorConfig()

    def extract_frames(
        self, video_path: Path, strategy: Optional[str] = None
    ) -> List[Frame]:
        """
        Extract frames from the video using the specified strategy.

        Args:
            video_path: Path to the video file.
            strategy: Strategy to use for frame extraction. If None, the strategy from the config is used.

        Returns:
            List[Frame]: List of extracted frames.

        Raises:
            VideoProcessingError: If there's an error processing the video.
            ValueError: If the strategy is not supported.
        """
        if strategy is None:
            strategy = self.config.strategy
        else:
            try:
                strategy = ExtractionStrategy(strategy)
            except ValueError:
                raise ValueError(
                    f"Unsupported extraction strategy: {strategy}. "
                    f"Supported strategies: {[s.value for s in ExtractionStrategy]}"
                )

        try:
            if strategy == ExtractionStrategy.UNIFORM:
                return self._extract_uniform(video_path)
            elif strategy == ExtractionStrategy.SCENE_CHANGE:
                return self._extract_scene_change(video_path)
            elif strategy == ExtractionStrategy.KEYFRAME:
                return self._extract_keyframes(video_path)
            else:
                raise ValueError(f"Unsupported extraction strategy: {strategy}")
        except Exception as e:
            raise VideoProcessingError(
                f"Error extracting frames: {str(e)}", details={"exception": str(e)}
            )

    def extract_frames_to_video_data(
        self, video_data: VideoData, strategy: Optional[str] = None
    ) -> VideoData:
        """
        Extract frames and add them to the VideoData object.

        Args:
            video_data: VideoData object to add frames to.
            strategy: Strategy to use for frame extraction. If None, the strategy from the config is used.

        Returns:
            VideoData: VideoData object with extracted frames.

        Raises:
            VideoProcessingError: If there's an error processing the video.
            ValueError: If the strategy is not supported.
        """
        frames = self.extract_frames(video_data.path, strategy)
        video_data.frames = frames
        return video_data

    def _extract_uniform(self, video_path: Path) -> List[Frame]:
        """
        Extract frames at uniform intervals.

        Args:
            video_path: Path to the video file.

        Returns:
            List[Frame]: List of extracted frames.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoProcessingError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0

        # Calculate frame indices to extract
        interval_frames = int(self.config.uniform_interval_seconds * fps)
        if interval_frames <= 0:
            interval_frames = 1

        frame_indices = list(range(0, frame_count, interval_frames))

        # Apply max_frames limit if specified
        if (
            self.config.max_frames is not None
            and len(frame_indices) > self.config.max_frames
        ):
            # Evenly distribute the frames across the video
            step = len(frame_indices) / self.config.max_frames
            frame_indices = [
                frame_indices[int(i * step)] for i in range(self.config.max_frames)
            ]

        frames = []

        # Process frames in batches for large videos
        for batch_indices in self._batch_iterator(
            frame_indices, self.config.batch_size
        ):
            batch_frames = self._extract_frame_batch(cap, batch_indices, fps)
            frames.extend(batch_frames)

        cap.release()
        return frames

    def _extract_scene_change(self, video_path: Path) -> List[Frame]:
        """
        Extract frames at scene changes.

        Args:
            video_path: Path to the video file.

        Returns:
            List[Frame]: List of extracted frames.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoProcessingError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize variables for scene change detection
        prev_frame = None
        frame_indices = []

        # Always include the first frame
        frame_indices.append(0)

        # Process frames to detect scene changes
        for i in range(0, frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale for faster processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                # Calculate difference between current and previous frame
                diff = cv2.absdiff(gray, prev_frame)
                non_zero_count = np.count_nonzero(diff)

                # If the difference is above threshold, consider it a scene change
                if non_zero_count > (
                    gray.shape[0]
                    * gray.shape[1]
                    * self.config.scene_change_threshold
                    / 100
                ):
                    frame_indices.append(i)

            prev_frame = gray

            # Check if we've reached the maximum number of frames
            if (
                self.config.max_frames is not None
                and len(frame_indices) >= self.config.max_frames
            ):
                break

        # Reset the video capture to extract the identified frames
        cap.release()
        cap = cv2.VideoCapture(str(video_path))

        frames = []

        # Process frames in batches for large videos
        for batch_indices in self._batch_iterator(
            frame_indices, self.config.batch_size
        ):
            batch_frames = self._extract_frame_batch(cap, batch_indices, fps)
            frames.extend(batch_frames)

        cap.release()
        return frames

    def _extract_keyframes(self, video_path: Path) -> List[Frame]:
        """
        Extract keyframes from the video.

        Args:
            video_path: Path to the video file.

        Returns:
            List[Frame]: List of extracted frames.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise VideoProcessingError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # For simplicity, we'll use I-frames as keyframes
        # In a real implementation, this would use more sophisticated methods

        frame_indices = []

        # Always include the first frame
        frame_indices.append(0)

        # OpenCV doesn't provide direct access to I-frames, so we'll use a heuristic
        # For H.264, I-frames typically occur every 12-15 frames
        # This is a simplified approach; a real implementation would use FFmpeg or similar
        i_frame_interval = 15

        for i in range(i_frame_interval, frame_count, i_frame_interval):
            frame_indices.append(i)

            # Check if we've reached the maximum number of frames
            if (
                self.config.max_frames is not None
                and len(frame_indices) >= self.config.max_frames
            ):
                break

        frames = []

        # Process frames in batches for large videos
        for batch_indices in self._batch_iterator(
            frame_indices, self.config.batch_size
        ):
            batch_frames = self._extract_frame_batch(cap, batch_indices, fps)
            frames.extend(batch_frames)

        cap.release()
        return frames

    def _extract_frame_batch(
        self, cap: cv2.VideoCapture, frame_indices: List[int], fps: float
    ) -> List[Frame]:
        """
        Extract a batch of frames from the video.

        Args:
            cap: OpenCV VideoCapture object.
            frame_indices: List of frame indices to extract.
            fps: Frames per second of the video.

        Returns:
            List[Frame]: List of extracted frames.
        """
        frames = []

        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        for idx in frame_indices:
            # If we need to move backward or if the jump is too large, seek to the position
            if idx < current_pos or idx > current_pos + 10:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            # Otherwise, read frames until we reach the desired position
            else:
                for _ in range(idx - current_pos):
                    cap.read()

            ret, image = cap.read()
            if not ret:
                logger.warning(f"Could not read frame at index {idx}")
                continue

            timestamp = idx / fps if fps > 0 else 0
            frames.append(Frame(image=image, timestamp=timestamp, index=idx))

            current_pos = idx + 1

        return frames

    def _batch_iterator(self, items: List[Any], batch_size: int) -> Iterator[List[Any]]:
        """
        Split a list into batches.

        Args:
            items: List to split into batches.
            batch_size: Size of each batch.

        Returns:
            Iterator[List[Any]]: Iterator over batches.
        """
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

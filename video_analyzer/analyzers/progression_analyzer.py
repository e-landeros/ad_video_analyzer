"""
Progression analyzer for video structure analysis.

This module provides functionality to analyze the structure and pacing of videos,
identifying sections, transitions, and evaluating narrative flow.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import asyncio
from datetime import datetime

from video_analyzer.analyzers.base import BaseAnalyzer, AnalyzerRegistry
from video_analyzer.models.video import VideoData, Frame
from video_analyzer.models.analysis import ProgressionAnalysisResult

# Set up logging
logger = logging.getLogger(__name__)

# Common retention strategies used in videos
RETENTION_STRATEGIES = [
    "pattern_interrupts",
    "open_loops",
    "curiosity_hooks",
    "storytelling",
    "pacing_variation",
    "emotional_triggers",
    "callbacks",
    "anticipation_building",
    "value_stacking",
    "direct_engagement",
]

# Common transition types in videos
TRANSITION_TYPES = [
    "cut",
    "fade",
    "dissolve",
    "wipe",
    "zoom",
    "slide",
    "visual_cue",
    "audio_cue",
    "topic_shift",
    "mood_change",
]


@AnalyzerRegistry.register("progression")
class ProgressionAnalyzer(BaseAnalyzer):
    """
    Analyzes the structure and pacing of videos.

    This analyzer breaks down videos into distinct sections, identifies pacing changes,
    detects transitions between topics or scenes, evaluates narrative flow, and
    provides insights on retention strategies.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the progression analyzer.

        Args:
            config: Optional configuration for the analyzer
        """
        super().__init__(config)

        # Default progression analysis parameters
        self.min_section_duration = self._config.get(
            "min_section_duration", 5.0
        )  # seconds
        self.transition_detection_threshold = self._config.get(
            "transition_detection_threshold", 40.0
        )
        self.scene_change_threshold = self._config.get("scene_change_threshold", 50.0)
        self.pacing_window_size = self._config.get(
            "pacing_window_size", 10.0
        )  # seconds

    @property
    def analyzer_id(self) -> str:
        """
        Get the unique identifier for this analyzer.

        Returns:
            str: The analyzer ID
        """
        return "progression_analyzer"

    @property
    def supports_progress(self) -> bool:
        """
        Check if this analyzer supports progress reporting.

        Returns:
            bool: True if progress reporting is supported
        """
        return True

    @property
    def required_frames(self) -> Dict[str, Any]:
        """
        Get the frame requirements for this analyzer.

        Returns:
            Dict[str, Any]: The frame requirements
        """
        return {
            "min_frames": 10,
            "frame_interval": 1.0,  # 1 frame per second is sufficient for progression analysis
            "specific_timestamps": None,
        }

    async def analyze(self, video_data: VideoData) -> ProgressionAnalysisResult:
        """
        Analyze the structure and pacing of the video.

        Args:
            video_data: The video data to analyze

        Returns:
            ProgressionAnalysisResult: The progression analysis result
        """
        logger.info(f"Analyzing progression for video: {video_data.path}")

        # Ensure we have enough frames to analyze
        if len(video_data.frames) < 10:
            logger.warning(
                f"Not enough frames for progression analysis: {video_data.path}"
            )
            # Create a minimal result with default values
            return ProgressionAnalysisResult(
                analyzer_id=self.analyzer_id,
                narrative_flow_score=0.5,  # Default middle value
                video_id=str(video_data.path),
                confidence=0.5,  # Low confidence due to lack of frames
            )

        # Identify video sections
        sections = await self._identify_sections(video_data)

        # Detect pacing changes
        pacing_changes = await self._detect_pacing_changes(video_data)

        # Identify transitions between topics or scenes
        transitions = await self._identify_transitions(video_data)

        # Evaluate narrative flow and coherence
        narrative_flow_score = await self._evaluate_narrative_flow(
            video_data, sections, transitions
        )

        # Identify retention strategies
        retention_strategies = await self._identify_retention_strategies(
            video_data, sections, pacing_changes
        )

        # Create the result
        result = ProgressionAnalysisResult(
            analyzer_id=self.analyzer_id,
            sections=sections,
            pacing_changes=pacing_changes,
            transitions=transitions,
            narrative_flow_score=narrative_flow_score,
            retention_strategies=retention_strategies,
            video_id=str(video_data.path),
            timestamps=[frame.timestamp for frame in video_data.frames],
            confidence=0.75,  # Reasonable confidence for progression analysis
        )

        logger.info(f"Progression analysis completed for video: {video_data.path}")
        return result

    async def _identify_sections(self, video_data: VideoData) -> List[Dict[str, Any]]:
        """
        Identify distinct sections or segments in the video.

        Args:
            video_data: The video data

        Returns:
            List[Dict[str, Any]]: Identified sections with timestamps
        """
        logger.debug("Identifying video sections")
        sections = []

        # If video is very short, treat it as a single section
        if video_data.duration < self.min_section_duration * 2:
            sections.append(
                {
                    "start_time": 0.0,
                    "end_time": video_data.duration,
                    "title": "Full Video",
                    "description": "Complete video content",
                    "type": "content",
                }
            )
            return sections

        # Analyze frame differences to detect significant changes that might indicate section boundaries
        frame_diffs = []
        for i in range(1, len(video_data.frames)):
            # Calculate mean absolute difference between consecutive frames
            diff = np.mean(
                np.abs(
                    video_data.frames[i].image.astype(float)
                    - video_data.frames[i - 1].image.astype(float)
                )
            )
            frame_diffs.append((video_data.frames[i].timestamp, diff))

        # Identify potential section boundaries based on significant frame differences
        potential_boundaries = []
        avg_diff = np.mean([d for _, d in frame_diffs])
        std_diff = np.std([d for _, d in frame_diffs])
        threshold = avg_diff + (std_diff * 2)  # 2 standard deviations above mean

        for timestamp, diff in frame_diffs:
            if diff > threshold:
                potential_boundaries.append(timestamp)

        # Filter boundaries to ensure minimum section duration
        filtered_boundaries = [0.0]  # Start with the beginning of the video
        for boundary in potential_boundaries:
            if boundary - filtered_boundaries[-1] >= self.min_section_duration:
                filtered_boundaries.append(boundary)
        filtered_boundaries.append(video_data.duration)  # End with the end of the video

        # Create sections from boundaries
        for i in range(len(filtered_boundaries) - 1):
            start_time = filtered_boundaries[i]
            end_time = filtered_boundaries[i + 1]

            # Generate a generic title and description
            section_number = i + 1
            section_type = self._determine_section_type(
                section_number, len(filtered_boundaries) - 1
            )

            sections.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "title": f"Section {section_number}",
                    "description": f"Video content from {start_time:.1f}s to {end_time:.1f}s",
                    "type": section_type,
                }
            )

        return sections

    def _determine_section_type(self, section_number: int, total_sections: int) -> str:
        """
        Determine the type of a section based on its position in the video.

        Args:
            section_number: The section number (1-based)
            total_sections: The total number of sections

        Returns:
            str: The section type
        """
        if section_number == 1:
            return "introduction"
        elif section_number == total_sections:
            return "conclusion"
        else:
            return "content"

    async def _detect_pacing_changes(
        self, video_data: VideoData
    ) -> List[Dict[str, Any]]:
        """
        Detect pacing changes throughout the video.

        Args:
            video_data: The video data

        Returns:
            List[Dict[str, Any]]: Detected pacing changes with timestamps
        """
        logger.debug("Detecting pacing changes")
        pacing_changes = []

        # If we don't have enough frames, return empty list
        if len(video_data.frames) < 10:
            return pacing_changes

        # Calculate frame differences to estimate motion/activity
        frame_diffs = []
        for i in range(1, len(video_data.frames)):
            # Calculate mean absolute difference between consecutive frames
            diff = np.mean(
                np.abs(
                    video_data.frames[i].image.astype(float)
                    - video_data.frames[i - 1].image.astype(float)
                )
            )
            frame_diffs.append((video_data.frames[i].timestamp, diff))

        # Use a sliding window to detect changes in pacing
        window_size = max(
            3, int(self.pacing_window_size / video_data.frames[1].timestamp)
        )

        for i in range(window_size, len(frame_diffs) - window_size):
            prev_window = frame_diffs[i - window_size : i]
            curr_window = frame_diffs[i : i + window_size]

            prev_avg = np.mean([d for _, d in prev_window])
            curr_avg = np.mean([d for _, d in curr_window])

            # If there's a significant change in the average difference, it might indicate a pacing change
            if abs(curr_avg - prev_avg) > (prev_avg * 0.5):  # 50% change threshold
                timestamp = frame_diffs[i][0]

                # Determine the type of pacing change
                if curr_avg > prev_avg:
                    pacing_type = "increase"
                    description = "Pacing increases, more activity or faster cuts"
                else:
                    pacing_type = "decrease"
                    description = "Pacing decreases, less activity or slower cuts"

                # Check if this change is close to a previously detected one
                if (
                    not pacing_changes
                    or abs(timestamp - pacing_changes[-1]["timestamp"])
                    > self.pacing_window_size
                ):
                    pacing_changes.append(
                        {
                            "timestamp": timestamp,
                            "type": pacing_type,
                            "description": description,
                            "magnitude": abs(curr_avg - prev_avg) / prev_avg,
                        }
                    )

        return pacing_changes

    async def _identify_transitions(
        self, video_data: VideoData
    ) -> List[Dict[str, Any]]:
        """
        Identify transitions between topics or scenes.

        Args:
            video_data: The video data

        Returns:
            List[Dict[str, Any]]: Identified transitions with timestamps
        """
        logger.debug("Identifying transitions")
        transitions = []

        # If we don't have enough frames, return empty list
        if len(video_data.frames) < 3:
            return transitions

        # Calculate frame differences to detect scene changes
        for i in range(1, len(video_data.frames)):
            # Calculate mean absolute difference between consecutive frames
            diff = np.mean(
                np.abs(
                    video_data.frames[i].image.astype(float)
                    - video_data.frames[i - 1].image.astype(float)
                )
            )

            # If the difference exceeds the threshold, it might be a scene change
            if diff > self.scene_change_threshold:
                # Determine transition type based on the characteristics of the change
                transition_type = self._determine_transition_type(
                    video_data.frames[i - 1], video_data.frames[i], diff
                )

                transitions.append(
                    {
                        "timestamp": video_data.frames[i].timestamp,
                        "type": transition_type,
                        "confidence": min(
                            1.0, diff / (self.scene_change_threshold * 2)
                        ),
                        "from_frame_index": video_data.frames[i - 1].index,
                        "to_frame_index": video_data.frames[i].index,
                    }
                )

        return transitions

    def _determine_transition_type(
        self, frame1: Frame, frame2: Frame, diff: float
    ) -> str:
        """
        Determine the type of transition between two frames.

        Args:
            frame1: The first frame
            frame2: The second frame
            diff: The difference between the frames

        Returns:
            str: The transition type
        """
        # This is a simplified implementation that would be enhanced in a real system

        # Check for fade by looking at overall brightness change
        brightness1 = np.mean(frame1.image)
        brightness2 = np.mean(frame2.image)

        if abs(brightness2 - brightness1) > 50:
            if brightness2 < brightness1:
                return "fade_out"
            else:
                return "fade_in"

        # Check for dissolve by looking at overall contrast
        contrast1 = np.std(frame1.image)
        contrast2 = np.std(frame2.image)

        if abs(contrast2 - contrast1) > 20:
            return "dissolve"

        # Default to cut for significant changes
        if diff > self.transition_detection_threshold * 2:
            return "cut"

        # For more subtle changes, call it a topic shift
        return "topic_shift"

    async def _evaluate_narrative_flow(
        self,
        video_data: VideoData,
        sections: List[Dict[str, Any]],
        transitions: List[Dict[str, Any]],
    ) -> float:
        """
        Evaluate the narrative flow and coherence of the video.

        Args:
            video_data: The video data
            sections: Identified video sections
            transitions: Identified transitions

        Returns:
            float: Narrative flow score (0.0-1.0)
        """
        logger.debug("Evaluating narrative flow")

        # Base flow score
        flow_score = 0.7  # Start with a reasonable default

        # Adjust based on number of sections relative to video length
        # Too many sections might indicate choppy flow, too few might be monotonous
        ideal_section_count = max(
            3, video_data.duration / 60
        )  # About one section per minute is reasonable
        actual_section_count = len(sections)

        section_ratio = min(actual_section_count, ideal_section_count) / max(
            actual_section_count, ideal_section_count
        )
        flow_score += (section_ratio - 0.5) * 0.2  # Adjust by up to ±0.1

        # Adjust based on transition distribution
        # Well-distributed transitions suggest better flow
        if transitions and len(sections) > 1:
            # Calculate ideal transition points (evenly distributed)
            ideal_transition_points = [
                sections[i]["end_time"] for i in range(len(sections) - 1)
            ]

            # Calculate how well actual transitions match ideal points
            if ideal_transition_points:
                transition_timestamps = [t["timestamp"] for t in transitions]

                # For each ideal point, find the closest actual transition
                total_deviation = 0
                for ideal_point in ideal_transition_points:
                    closest_transition = min(
                        transition_timestamps, key=lambda t: abs(t - ideal_point)
                    )
                    deviation = (
                        abs(closest_transition - ideal_point) / video_data.duration
                    )
                    total_deviation += deviation

                avg_deviation = total_deviation / len(ideal_transition_points)
                flow_score -= (
                    avg_deviation * 0.5
                )  # Penalize by up to 0.5 for poor transition placement

        # Adjust based on section duration consistency
        # Consistent section durations might indicate better planning and flow
        if len(sections) > 1:
            section_durations = [s["end_time"] - s["start_time"] for s in sections]
            avg_duration = sum(section_durations) / len(section_durations)
            duration_variance = sum(
                (d - avg_duration) ** 2 for d in section_durations
            ) / len(section_durations)
            normalized_variance = min(1.0, duration_variance / (avg_duration**2))

            flow_score -= (
                normalized_variance * 0.2
            )  # Penalize by up to 0.2 for inconsistent section durations

        # Ensure the score is within bounds
        flow_score = max(0.0, min(flow_score, 1.0))

        return flow_score

    async def _identify_retention_strategies(
        self,
        video_data: VideoData,
        sections: List[Dict[str, Any]],
        pacing_changes: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Identify retention strategies used in the video.

        Args:
            video_data: The video data
            sections: Identified video sections
            pacing_changes: Detected pacing changes

        Returns:
            List[str]: Identified retention strategies
        """
        logger.debug("Identifying retention strategies")
        strategies = []

        # Check for pattern interrupts based on pacing changes
        if len(pacing_changes) > 2:
            strategies.append("pattern_interrupts")

        # Check for pacing variation
        if len(pacing_changes) > 0:
            strategies.append("pacing_variation")

        # Check for storytelling based on section structure
        if len(sections) >= 3:
            intro_section = next(
                (s for s in sections if s["type"] == "introduction"), None
            )
            conclusion_section = next(
                (s for s in sections if s["type"] == "conclusion"), None
            )

            if intro_section and conclusion_section:
                strategies.append("storytelling")

        # Check for open loops based on section count and video duration
        if (
            len(sections) > 3 and video_data.duration > 120
        ):  # Longer videos with multiple sections
            strategies.append("open_loops")

        # Add value stacking for videos with many sections
        if len(sections) > 5:
            strategies.append("value_stacking")

        # If we couldn't identify specific strategies, add a default
        if not strategies and video_data.duration > 30:
            strategies.append("direct_engagement")  # Most common strategy

        return strategies

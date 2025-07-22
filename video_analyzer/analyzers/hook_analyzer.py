"""
Hook analyzer for video introductions.

This module provides functionality to analyze the hook section of videos,
identifying techniques used to capture viewer attention and evaluating
their effectiveness.
"""

import logging
from typing import Dict, Any, List
import numpy as np
import openai
import os

from video_analyzer.analyzers.base import BaseAnalyzer, AnalyzerRegistry
from video_analyzer.models.video import VideoData
from video_analyzer.models.analysis import HookAnalysisResult

# Set up logging
logger = logging.getLogger(__name__)

# Hook techniques commonly used in videos
HOOK_TECHNIQUES = [
    "question",
    "shocking_statement",
    "story",
    "preview",
    "challenge",
    "curiosity_gap",
    "direct_address",
    "quote",
    "statistic",
    "music",
    "visual_pattern_interrupt",
    "problem_solution",
    "testimonial",
    "demonstration",
    "humor",
]


@AnalyzerRegistry.register("hook")
class HookAnalyzer(BaseAnalyzer):
    """
    Analyzes the hook section of videos.

    This analyzer identifies the hook section (typically the first 5-15 seconds),
    describes techniques used to capture attention, evaluates effectiveness,
    and provides timestamps for key moments.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the hook analyzer.

        Args:
            config: Optional configuration for the analyzer
        """
        super().__init__(config)
        self.openai_api_key = self._config.get("openai_api_key") or os.environ.get(
            "OPENAI_API_KEY"
        )
        self.openai_client = None
        if self.openai_api_key:
            self.openai_client = openai.AsyncOpenAI(api_key=self.openai_api_key)

        # Default hook analysis parameters
        self.default_hook_duration = self._config.get(
            "default_hook_duration", 15
        )  # seconds
        self.min_hook_duration = self._config.get("min_hook_duration", 3)  # seconds
        self.max_hook_duration = self._config.get("max_hook_duration", 30)  # seconds

    @property
    def analyzer_id(self) -> str:
        """
        Get the unique identifier for this analyzer.

        Returns:
            str: The analyzer ID
        """
        return "hook_analyzer"

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
            "min_frames": 5,
            "frame_interval": 1.0,  # 1 frame per second is sufficient for hook analysis
            "specific_timestamps": None,
        }

    async def analyze(self, video_data: VideoData) -> HookAnalysisResult:
        """
        Analyze the hook section of the video.

        Args:
            video_data: The video data to analyze

        Returns:
            HookAnalysisResult: The hook analysis result
        """
        logger.info(f"Analyzing hook for video: {video_data.path}")

        # Determine the hook section (typically first 5-15 seconds)
        hook_end_time = min(self.default_hook_duration, video_data.duration)
        hook_start_time = 0

        # Get frames in the hook section
        hook_frames = video_data.get_frames_in_range(hook_start_time, hook_end_time)

        if not hook_frames:
            logger.warning(
                f"No frames found in hook section for video: {video_data.path}"
            )
            # Create a minimal result with default values
            return HookAnalysisResult(
                analyzer_id=self.analyzer_id,
                hook_start_time=hook_start_time,
                hook_end_time=hook_end_time,
                hook_effectiveness=0.5,  # Default middle value
                hook_techniques=["unknown"],
                video_id=str(video_data.path),
                confidence=0.5,  # Low confidence due to lack of frames
            )

        # Analyze the hook section
        hook_techniques = await self._identify_hook_techniques(video_data, hook_frames)
        hook_effectiveness = await self._evaluate_hook_effectiveness(
            video_data, hook_frames, hook_techniques
        )
        key_moments = await self._identify_key_moments(video_data, hook_frames)
        recommendations = await self._generate_recommendations(
            video_data, hook_techniques, hook_effectiveness
        )

        # Create the result
        result = HookAnalysisResult(
            analyzer_id=self.analyzer_id,
            hook_start_time=hook_start_time,
            hook_end_time=hook_end_time,
            hook_techniques=hook_techniques,
            hook_effectiveness=hook_effectiveness,
            key_moments=key_moments,
            recommendations=recommendations,
            video_id=str(video_data.path),
            timestamps=[frame.timestamp for frame in hook_frames],
            confidence=0.8,  # Reasonable confidence for hook analysis
        )

        logger.info(f"Hook analysis completed for video: {video_data.path}")
        return result

    async def _identify_hook_techniques(
        self, video_data: VideoData, hook_frames: List
    ) -> List[str]:
        """
        Identify techniques used in the hook section.

        Args:
            video_data: The video data
            hook_frames: Frames in the hook section

        Returns:
            List[str]: Identified hook techniques
        """
        # If we have OpenAI integration, use it for more sophisticated analysis
        if self.openai_client:
            return await self._identify_hook_techniques_with_llm(
                video_data, hook_frames
            )

        # Fallback to basic analysis
        return self._identify_hook_techniques_basic(video_data, hook_frames)

    async def _identify_hook_techniques_with_llm(
        self, video_data: VideoData, hook_frames: List
    ) -> List[str]:
        """
        Use OpenAI LLM to identify hook techniques.

        Args:
            video_data: The video data
            hook_frames: Frames in the hook section

        Returns:
            List[str]: Identified hook techniques
        """
        try:
            # Prepare frame descriptions for the LLM
            frame_descriptions = []
            for i, frame in enumerate(
                hook_frames[:5]
            ):  # Limit to first 5 frames to avoid token limits
                # Simple frame description based on basic image properties
                avg_color = np.mean(frame.image, axis=(0, 1))
                brightness = np.mean(frame.image)
                frame_descriptions.append(
                    f"Frame {i + 1} at {frame.timestamp:.2f}s: "
                    f"Brightness: {brightness:.2f}, "
                    f"Average RGB: ({avg_color[0]:.2f}, {avg_color[1]:.2f}, {avg_color[2]:.2f})"
                )

            # Create prompt for the LLM
            # Create prompt parts separately
            hook_timestamp = hook_frames[-1].timestamp
            resolution_str = f"{video_data.resolution[0]}x{video_data.resolution[1]}"
            frame_desc_str = "\n".join(frame_descriptions)
            techniques_str = ", ".join(HOOK_TECHNIQUES)

            # Combine into prompt
            prompt = f"""
            Analyze the following video hook section (first {hook_timestamp:.2f} seconds):
            
            Video metadata:
            - Duration: {video_data.duration} seconds
            - Resolution: {resolution_str}
            
            Frame descriptions:
            {frame_desc_str}
            
            Based on this information, identify which of the following hook techniques are likely used:
            {techniques_str}
            
            Return only the technique names as a comma-separated list.
            """

            # Call OpenAI API
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a video analysis expert specializing in identifying hook techniques in video introductions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=100,
                temperature=0.3,
            )

            # Parse the response
            techniques_text = response.choices[0].message.content.strip()
            techniques = [tech.strip().lower() for tech in techniques_text.split(",")]

            # Filter to ensure only valid techniques are included
            valid_techniques = [
                tech for tech in techniques if any(vt in tech for vt in HOOK_TECHNIQUES)
            ]

            # If no valid techniques found, fall back to basic analysis
            if not valid_techniques:
                logger.warning(
                    "No valid hook techniques identified by LLM, falling back to basic analysis"
                )
                return self._identify_hook_techniques_basic(video_data, hook_frames)

            return valid_techniques

        except Exception as e:
            logger.error(
                f"Error using OpenAI for hook technique identification: {str(e)}",
                exc_info=True,
            )
            # Fall back to basic analysis
            return self._identify_hook_techniques_basic(video_data, hook_frames)

    def _identify_hook_techniques_basic(
        self, video_data: VideoData, hook_frames: List
    ) -> List[str]:
        """
        Use basic heuristics to identify hook techniques.

        Args:
            video_data: The video data
            hook_frames: Frames in the hook section

        Returns:
            List[str]: Identified hook techniques
        """
        # This is a simplified implementation that would be enhanced in a real system
        # with more sophisticated image and audio analysis

        techniques = []

        # Check for visual pattern interrupts (significant changes between frames)
        if len(hook_frames) > 1:
            frame_diffs = []
            for i in range(1, len(hook_frames)):
                # Calculate mean absolute difference between consecutive frames
                diff = np.mean(
                    np.abs(
                        hook_frames[i].image.astype(float)
                        - hook_frames[i - 1].image.astype(float)
                    )
                )
                frame_diffs.append(diff)

            # If we have significant differences between frames, it might be a visual pattern interrupt
            if (
                frame_diffs and max(frame_diffs) > 30
            ):  # Threshold determined empirically
                techniques.append("visual_pattern_interrupt")

        # Add some default techniques based on video properties
        if video_data.duration > 60:  # Longer videos often use preview hooks
            techniques.append("preview")

        # If we couldn't identify any techniques, add a default
        if not techniques:
            techniques.append("direct_address")  # Most common technique

        return techniques

    async def _evaluate_hook_effectiveness(
        self, video_data: VideoData, hook_frames: List, hook_techniques: List[str]
    ) -> float:
        """
        Evaluate the effectiveness of the hook.

        Args:
            video_data: The video data
            hook_frames: Frames in the hook section
            hook_techniques: Identified hook techniques

        Returns:
            float: Hook effectiveness score (0.0-1.0)
        """
        # This is a simplified implementation that would be enhanced in a real system

        # Base effectiveness score
        effectiveness = 0.7  # Start with a reasonable default

        # Adjust based on number of techniques used (more techniques might be more engaging)
        technique_factor = min(len(hook_techniques) / 3, 1.0)  # Cap at 1.0
        effectiveness += technique_factor * 0.1

        # Adjust based on hook duration relative to video length
        hook_duration = hook_frames[-1].timestamp - hook_frames[0].timestamp
        duration_ratio = hook_duration / video_data.duration

        # Ideal hook duration is 5-10% of total video length
        if 0.05 <= duration_ratio <= 0.1:
            effectiveness += 0.1
        elif duration_ratio > 0.2:  # Too long
            effectiveness -= 0.1

        # Ensure the score is within bounds
        effectiveness = max(0.0, min(effectiveness, 1.0))

        return effectiveness

    async def _identify_key_moments(
        self, video_data: VideoData, hook_frames: List
    ) -> List[Dict[str, Any]]:
        """
        Identify key moments in the hook section.

        Args:
            video_data: The video data
            hook_frames: Frames in the hook section

        Returns:
            List[Dict[str, Any]]: Key moments with timestamps
        """
        key_moments = []

        # This is a simplified implementation that would be enhanced in a real system

        # Consider the first frame as a key moment (hook start)
        if hook_frames:
            key_moments.append(
                {
                    "timestamp": hook_frames[0].timestamp,
                    "description": "Hook start",
                    "importance": 0.9,
                }
            )

        # If we have enough frames, identify moments with significant visual changes
        if len(hook_frames) > 2:
            for i in range(1, len(hook_frames) - 1):
                # Calculate difference from previous and next frames
                prev_diff = np.mean(
                    np.abs(
                        hook_frames[i].image.astype(float)
                        - hook_frames[i - 1].image.astype(float)
                    )
                )
                next_diff = np.mean(
                    np.abs(
                        hook_frames[i].image.astype(float)
                        - hook_frames[i + 1].image.astype(float)
                    )
                )

                # If this frame is significantly different from both neighbors, it might be a key moment
                if (
                    prev_diff > 30 and next_diff > 30
                ):  # Threshold determined empirically
                    key_moments.append(
                        {
                            "timestamp": hook_frames[i].timestamp,
                            "description": "Visual change",
                            "importance": 0.7,
                        }
                    )

        # Consider the last frame as a key moment (hook end)
        if hook_frames:
            key_moments.append(
                {
                    "timestamp": hook_frames[-1].timestamp,
                    "description": "Hook end",
                    "importance": 0.8,
                }
            )

        return key_moments

    async def _generate_recommendations(
        self,
        video_data: VideoData,
        hook_techniques: List[str],
        hook_effectiveness: float,
    ) -> List[str]:
        """
        Generate recommendations for improving the hook.

        Args:
            video_data: The video data
            hook_techniques: Identified hook techniques
            hook_effectiveness: Hook effectiveness score

        Returns:
            List[str]: Recommendations
        """
        recommendations = []

        # This is a simplified implementation that would be enhanced in a real system

        # If effectiveness is low, suggest improvements
        if hook_effectiveness < 0.6:
            recommendations.append("Consider using more engaging hook techniques")

            # Suggest techniques not currently used
            unused_techniques = [
                tech for tech in HOOK_TECHNIQUES if tech not in hook_techniques
            ]
            if unused_techniques:
                # Recommend a few unused techniques
                suggested_techniques = unused_techniques[:3]
                recommendations.append(
                    f"Try incorporating these techniques: {', '.join(suggested_techniques)}"
                )

        # If hook is too short or too long relative to video length
        hook_duration = (
            video_data.get_frames_in_range(0, self.default_hook_duration)[-1].timestamp
            if video_data.frames
            else self.default_hook_duration
        )
        duration_ratio = hook_duration / video_data.duration

        if duration_ratio < 0.03:
            recommendations.append(
                "Consider extending the hook to better engage viewers"
            )
        elif duration_ratio > 0.15:
            recommendations.append(
                "Consider shortening the hook to maintain viewer interest"
            )

        # If no recommendations, add a positive note
        if not recommendations:
            recommendations.append("The hook is effective and well-constructed")

        return recommendations

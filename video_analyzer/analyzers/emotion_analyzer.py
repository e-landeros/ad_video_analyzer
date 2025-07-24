"""
Emotion analyzer for video mood and emotional impact.

This module provides functionality to analyze the emotional aspects of videos,
identifying overall mood, emotional shifts, and creating emotional journey maps.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import openai
import os
import asyncio

from video_analyzer.analyzers.base import BaseAnalyzer, AnalyzerRegistry
from video_analyzer.models.video import VideoData
from video_analyzer.models.analysis import EmotionAnalysisResult

# Set up logging
logger = logging.getLogger(__name__)

# Common emotions in videos
COMMON_EMOTIONS = [
    "joy",
    "sadness",
    "fear",
    "anger",
    "surprise",
    "disgust",
    "anticipation",
    "trust",
    "excitement",
    "calm",
    "tension",
    "nostalgia",
    "awe",
    "confusion",
    "amusement",
    "inspiration",
]

# Common emotional triggers in videos
EMOTIONAL_TRIGGERS = [
    "music_change",
    "visual_contrast",
    "character_reaction",
    "narrative_twist",
    "pacing_change",
    "lighting_shift",
    "sound_effect",
    "dialogue",
    "silence",
    "color_palette_shift",
    "camera_movement",
]


@AnalyzerRegistry.register("emotion")
class EmotionAnalyzer(BaseAnalyzer):
    """
    Analyzes the emotional aspects of videos.

    This analyzer identifies the overall mood/tone, detects emotional shifts,
    evaluates how visual and audio elements contribute to emotions, identifies
    techniques used to elicit specific emotions, and creates an emotional journey map.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the emotion analyzer.

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

        # Default emotion analysis parameters
        self.sample_interval = self._config.get("sample_interval", 10)  # seconds
        self.min_emotion_shift_threshold = self._config.get(
            "min_emotion_shift_threshold", 0.3
        )

    @property
    def analyzer_id(self) -> str:
        """
        Get the unique identifier for this analyzer.

        Returns:
            str: The analyzer ID
        """
        return "emotion_analyzer"

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
            "frame_interval": self.sample_interval,  # Sample frames at regular intervals
            "specific_timestamps": None,
        }

    async def analyze(self, video_data: VideoData) -> EmotionAnalysisResult:
        """
        Analyze the emotional aspects of the video.

        Args:
            video_data: The video data to analyze

        Returns:
            EmotionAnalysisResult: The emotion analysis result
        """
        logger.info(f"Analyzing emotions for video: {video_data.path}")

        # Get frames at regular intervals for emotion analysis
        frames = video_data.get_frames_at_intervals(self.sample_interval)

        if not frames:
            logger.warning(
                f"No frames found for emotion analysis in video: {video_data.path}"
            )
            # Create a minimal result with default values
            return EmotionAnalysisResult(
                analyzer_id=self.analyzer_id,
                overall_mood="neutral",
                emotional_shifts=[],
                emotional_elements={"visual": [], "audio": []},
                emotion_techniques=[],
                emotional_journey=[],
                video_id=str(video_data.path),
                confidence=0.5,  # Low confidence due to lack of frames
            )

        # Analyze the emotional aspects of the video
        overall_mood = await self._identify_overall_mood(video_data, frames)
        emotional_journey = await self._create_emotional_journey(video_data, frames)
        emotional_shifts = await self._detect_emotional_shifts(emotional_journey)
        emotional_elements = await self._identify_emotional_elements(video_data, frames)
        emotion_techniques = await self._identify_emotion_techniques(
            video_data, frames, emotional_journey
        )

        # Create the result
        result = EmotionAnalysisResult(
            analyzer_id=self.analyzer_id,
            overall_mood=overall_mood,
            emotional_shifts=emotional_shifts,
            emotional_elements=emotional_elements,
            emotion_techniques=emotion_techniques,
            emotional_journey=emotional_journey,
            video_id=str(video_data.path),
            timestamps=[frame.timestamp for frame in frames],
            confidence=0.8,  # Reasonable confidence for emotion analysis
        )

        logger.info(f"Emotion analysis completed for video: {video_data.path}")
        return result

    async def _identify_overall_mood(self, video_data: VideoData, frames: List) -> str:
        """
        Identify the overall mood/tone of the video.

        Args:
            video_data: The video data
            frames: Sampled frames for analysis

        Returns:
            str: Overall mood/tone
        """
        # If we have OpenAI integration, use it for more sophisticated analysis
        if self.openai_client:
            return await self._identify_overall_mood_with_llm(video_data, frames)

        # Fallback to basic analysis
        return self._identify_overall_mood_basic(video_data, frames)

    async def _identify_overall_mood_with_llm(
        self, video_data: VideoData, frames: List
    ) -> str:
        """
        Use OpenAI LLM to identify the overall mood/tone.

        Args:
            video_data: The video data
            frames: Sampled frames for analysis

        Returns:
            str: Overall mood/tone
        """
        try:
            # Prepare frame descriptions for the LLM
            frame_descriptions = []
            for i, frame in enumerate(
                frames[:5]
            ):  # Limit to first 5 frames to avoid token limits
                # Simple frame description based on basic image properties
                avg_color = np.mean(frame.image, axis=(0, 1))
                brightness = np.mean(frame.image)
                saturation = np.std(frame.image, axis=(0, 1)).mean()
                frame_descriptions.append(
                    f"Frame {i + 1} at {frame.timestamp:.2f}s: "
                    f"Brightness: {brightness:.2f}, "
                    f"Saturation: {saturation:.2f}, "
                    f"Average RGB: ({avg_color[0]:.2f}, {avg_color[1]:.2f}, {avg_color[2]:.2f})"
                )

            # Create prompt for the LLM
            resolution_str = f"{video_data.resolution[0]}x{video_data.resolution[1]}"
            frame_desc_str = "\n".join(frame_descriptions)
            emotions_str = ", ".join(COMMON_EMOTIONS)

            # Combine into prompt
            prompt = f"""
            Analyze the following video frames to determine the overall mood/tone:
            
            Video metadata:
            - Duration: {video_data.duration} seconds
            - Resolution: {resolution_str}
            
            Frame descriptions:
            {frame_desc_str}
            
            Based on this information, identify the overall mood/tone of the video.
            Choose from these common emotions: {emotions_str}
            
            Return only the most dominant mood/tone as a single word.
            """

            # Call OpenAI API
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a video analysis expert specializing in identifying emotional tones in videos.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=50,
                temperature=0.3,
            )

            # Parse the response
            mood = response.choices[0].message.content.strip().lower()

            # Ensure the mood is one of the common emotions
            if not any(emotion in mood for emotion in COMMON_EMOTIONS):
                logger.warning(
                    f"LLM returned uncommon mood '{mood}', falling back to basic analysis"
                )
                return self._identify_overall_mood_basic(video_data, frames)

            return mood

        except Exception as e:
            logger.error(
                f"Error using OpenAI for mood identification: {str(e)}",
                exc_info=True,
            )
            # Fall back to basic analysis
            return self._identify_overall_mood_basic(video_data, frames)

    def _identify_overall_mood_basic(self, video_data: VideoData, frames: List) -> str:
        """
        Use basic heuristics to identify the overall mood/tone.

        Args:
            video_data: The video data
            frames: Sampled frames for analysis

        Returns:
            str: Overall mood/tone
        """
        # This is a simplified implementation that would be enhanced in a real system
        # with more sophisticated image and audio analysis

        # Calculate average brightness and saturation across frames
        brightness_values = []
        saturation_values = []

        for frame in frames:
            brightness = np.mean(frame.image)
            saturation = np.std(frame.image, axis=(0, 1)).mean()
            brightness_values.append(brightness)
            saturation_values.append(saturation)

        avg_brightness = np.mean(brightness_values) if brightness_values else 0
        avg_saturation = np.mean(saturation_values) if saturation_values else 0

        # Determine mood based on brightness and saturation
        # This is a very simplified approach
        if avg_brightness > 150 and avg_saturation > 50:
            return "joy"
        elif avg_brightness > 150 and avg_saturation <= 50:
            return "calm"
        elif avg_brightness <= 150 and avg_saturation > 50:
            return "tension"
        else:
            return "sadness"

    async def _create_emotional_journey(
        self, video_data: VideoData, frames: List
    ) -> List[Dict[str, Any]]:
        """
        Create an emotional journey map of the video.

        Args:
            video_data: The video data
            frames: Sampled frames for analysis

        Returns:
            List[Dict[str, Any]]: Emotional journey map
        """
        emotional_journey = []

        # If we have OpenAI integration, use it for more sophisticated analysis
        if self.openai_client:
            try:
                return await self._create_emotional_journey_with_llm(video_data, frames)
            except Exception as e:
                logger.error(
                    f"Error using OpenAI for emotional journey mapping: {str(e)}",
                    exc_info=True,
                )
                # Fall back to basic analysis

        # Basic analysis - analyze each frame individually
        for i, frame in enumerate(frames):
            # Simple emotion detection based on frame properties
            brightness = np.mean(frame.image)
            saturation = np.std(frame.image, axis=(0, 1)).mean()

            # Determine emotion based on brightness and saturation
            # This is a very simplified approach
            emotion = "neutral"
            intensity = 0.5  # Default middle value

            if brightness > 200:
                if saturation > 70:
                    emotion = "joy"
                    intensity = min(0.5 + (saturation - 70) / 100, 1.0)
                else:
                    emotion = "calm"
                    intensity = max(0.5 - (70 - saturation) / 100, 0.0)
            elif brightness < 100:
                if saturation > 70:
                    emotion = "anger"
                    intensity = min(0.5 + (saturation - 70) / 100, 1.0)
                else:
                    emotion = "sadness"
                    intensity = max(0.5 - (70 - saturation) / 100, 0.0)
            else:
                if saturation > 70:
                    emotion = "excitement"
                    intensity = min(0.5 + (saturation - 70) / 100, 1.0)
                else:
                    emotion = "neutral"
                    intensity = 0.5

            emotional_journey.append(
                {
                    "timestamp": frame.timestamp,
                    "emotion": emotion,
                    "intensity": intensity,
                }
            )

        return emotional_journey

    async def _create_emotional_journey_with_llm(
        self, video_data: VideoData, frames: List
    ) -> List[Dict[str, Any]]:
        """
        Use OpenAI LLM to create an emotional journey map.

        Args:
            video_data: The video data
            frames: Sampled frames for analysis

        Returns:
            List[Dict[str, Any]]: Emotional journey map
        """
        emotional_journey = []

        # Process frames in batches to avoid token limits
        batch_size = 5
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]

            # Prepare frame descriptions
            frame_descriptions = []
            for j, frame in enumerate(batch_frames):
                avg_color = np.mean(frame.image, axis=(0, 1))
                brightness = np.mean(frame.image)
                saturation = np.std(frame.image, axis=(0, 1)).mean()
                frame_descriptions.append(
                    f"Frame {i + j + 1} at {frame.timestamp:.2f}s: "
                    f"Brightness: {brightness:.2f}, "
                    f"Saturation: {saturation:.2f}, "
                    f"Average RGB: ({avg_color[0]:.2f}, {avg_color[1]:.2f}, {avg_color[2]:.2f})"
                )

            # Create prompt for the LLM
            frame_desc_str = "\n".join(frame_descriptions)
            emotions_str = ", ".join(COMMON_EMOTIONS)

            prompt = f"""
            Analyze the following video frames to determine the emotional journey:
            
            Frame descriptions:
            {frame_desc_str}
            
            For each frame, identify:
            1. The dominant emotion (choose from: {emotions_str})
            2. The intensity of that emotion (a value between 0.0 and 1.0)
            
            Return your analysis in this exact format, one line per frame:
            timestamp,emotion,intensity
            
            Example:
            5.20,joy,0.8
            15.50,sadness,0.6
            """

            # Call OpenAI API
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a video analysis expert specializing in identifying emotions in video frames.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.3,
            )

            # Parse the response
            lines = response.choices[0].message.content.strip().split("\n")
            for line in lines:
                if not line or "," not in line:
                    continue

                try:
                    parts = line.split(",")
                    if len(parts) != 3:
                        continue

                    timestamp_str, emotion, intensity_str = parts
                    timestamp = float(timestamp_str)
                    intensity = float(intensity_str)

                    # Validate values
                    if not 0 <= intensity <= 1:
                        intensity = max(0.0, min(intensity, 1.0))

                    emotional_journey.append(
                        {
                            "timestamp": timestamp,
                            "emotion": emotion.strip().lower(),
                            "intensity": intensity,
                        }
                    )
                except (ValueError, IndexError) as e:
                    logger.warning(
                        f"Error parsing emotional journey line '{line}': {str(e)}"
                    )
                    continue

            # Add a small delay to avoid rate limiting
            await asyncio.sleep(0.5)

        # Sort by timestamp
        emotional_journey.sort(key=lambda x: x["timestamp"])
        return emotional_journey

    async def _detect_emotional_shifts(
        self, emotional_journey: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect emotional shifts throughout the video.

        Args:
            emotional_journey: Emotional journey map

        Returns:
            List[Dict[str, Any]]: Emotional shifts
        """
        emotional_shifts = []

        # Need at least two points to detect shifts
        if len(emotional_journey) < 2:
            return emotional_shifts

        # Look for changes in emotion
        for i in range(1, len(emotional_journey)):
            current = emotional_journey[i]
            previous = emotional_journey[i - 1]

            # Check if emotion changed
            if current["emotion"] != previous["emotion"]:
                # Check if intensity changed significantly
                intensity_change = abs(current["intensity"] - previous["intensity"])

                # Only consider it a shift if the intensity change is significant
                if intensity_change >= self.min_emotion_shift_threshold:
                    # Determine the trigger (simplified approach)
                    trigger = self._determine_shift_trigger(previous, current)

                    emotional_shifts.append(
                        {
                            "timestamp": current["timestamp"],
                            "from_emotion": previous["emotion"],
                            "to_emotion": current["emotion"],
                            "intensity_change": intensity_change,
                            "trigger": trigger,
                        }
                    )

        return emotional_shifts

    def _determine_shift_trigger(
        self, previous: Dict[str, Any], current: Dict[str, Any]
    ) -> str:
        """
        Determine the trigger for an emotional shift.

        Args:
            previous: Previous emotional state
            current: Current emotional state

        Returns:
            str: Trigger for the emotional shift
        """
        # This is a simplified implementation
        # In a real system, we would analyze the frames and audio around the shift

        # For now, just return a random trigger
        import random

        return random.choice(EMOTIONAL_TRIGGERS)

    async def _identify_emotional_elements(
        self, video_data: VideoData, frames: List
    ) -> Dict[str, Any]:
        """
        Identify visual and audio elements contributing to emotions.

        Args:
            video_data: The video data
            frames: Sampled frames for analysis

        Returns:
            Dict[str, Any]: Emotional elements
        """
        # This is a simplified implementation

        visual_elements = []
        audio_elements = []

        # Analyze visual elements (simplified)
        for i, frame in enumerate(frames):
            # Only analyze a subset of frames to avoid redundancy
            if i % 3 != 0:
                continue

            brightness = np.mean(frame.image)
            saturation = np.std(frame.image, axis=(0, 1)).mean()

            # Identify visual elements based on simple properties
            if brightness > 200:
                visual_elements.append(
                    {
                        "timestamp": frame.timestamp,
                        "type": "high_brightness",
                        "description": "Bright lighting",
                        "emotional_impact": "positive"
                        if saturation > 50
                        else "neutral",
                    }
                )
            elif brightness < 100:
                visual_elements.append(
                    {
                        "timestamp": frame.timestamp,
                        "type": "low_brightness",
                        "description": "Dark lighting",
                        "emotional_impact": "negative"
                        if saturation > 50
                        else "mysterious",
                    }
                )

            if saturation > 70:
                visual_elements.append(
                    {
                        "timestamp": frame.timestamp,
                        "type": "high_saturation",
                        "description": "Vibrant colors",
                        "emotional_impact": "energetic",
                    }
                )
            elif saturation < 30:
                visual_elements.append(
                    {
                        "timestamp": frame.timestamp,
                        "type": "low_saturation",
                        "description": "Muted colors",
                        "emotional_impact": "somber",
                    }
                )

        # Audio elements would require audio analysis
        # This is a placeholder for a real implementation
        audio_elements = [
            {
                "timestamp": 0,
                "type": "background_music",
                "description": "Background music",
                "emotional_impact": "sets_mood",
            }
        ]

        return {
            "visual": visual_elements,
            "audio": audio_elements,
        }

    async def _identify_emotion_techniques(
        self,
        video_data: VideoData,
        frames: List,
        emotional_journey: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Identify techniques used to elicit specific emotions.

        Args:
            video_data: The video data
            frames: Sampled frames for analysis
            emotional_journey: Emotional journey map

        Returns:
            List[str]: Emotion techniques
        """
        # This is a simplified implementation

        techniques = []

        # Look for patterns in the emotional journey
        if len(emotional_journey) > 2:
            # Check for emotional contrast (rapid shifts between emotions)
            has_contrast = False
            for i in range(1, len(emotional_journey) - 1):
                if (
                    emotional_journey[i]["emotion"]
                    != emotional_journey[i - 1]["emotion"]
                    and emotional_journey[i]["emotion"]
                    != emotional_journey[i + 1]["emotion"]
                ):
                    has_contrast = True
                    break

            if has_contrast:
                techniques.append("emotional_contrast")

            # Check for emotional buildup (increasing intensity)
            intensities = [entry["intensity"] for entry in emotional_journey]
            if len(intensities) > 3:
                # Check if intensities are generally increasing
                increasing = all(
                    intensities[i] <= intensities[i + 1]
                    for i in range(len(intensities) - 1)
                )
                if increasing:
                    techniques.append("emotional_buildup")

            # Check for emotional release (high intensity followed by low)
            for i in range(1, len(emotional_journey)):
                if (
                    emotional_journey[i - 1]["intensity"] > 0.7
                    and emotional_journey[i]["intensity"] < 0.3
                ):
                    techniques.append("emotional_release")
                    break

        # Add some default techniques based on overall mood
        overall_mood = await self._identify_overall_mood(video_data, frames)

        if overall_mood in ["joy", "excitement"]:
            techniques.append("positive_reinforcement")
        elif overall_mood in ["sadness", "fear"]:
            techniques.append("empathy_building")
        elif overall_mood in ["anger", "disgust"]:
            techniques.append("moral_outrage")
        elif overall_mood in ["surprise", "awe"]:
            techniques.append("expectation_subversion")

        # Ensure we have at least one technique
        if not techniques:
            techniques.append("mood_setting")

        return techniques

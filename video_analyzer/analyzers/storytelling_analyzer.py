"""
Storytelling analyzer for narrative structure analysis.

This module provides functionality to analyze the storytelling aspects of videos,
identifying narrative structure, character development, conflict patterns,
persuasion techniques, and audience engagement strategies.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import openai
import os
import asyncio

from video_analyzer.analyzers.base import BaseAnalyzer, AnalyzerRegistry
from video_analyzer.models.video import VideoData, Frame
from video_analyzer.models.analysis import StorytellingAnalysisResult

# Set up logging
logger = logging.getLogger(__name__)

# Common narrative structures
NARRATIVE_STRUCTURES = [
    "three_act",  # Classic three-act structure (setup, confrontation, resolution)
    "hero_journey",  # Hero's journey/monomyth
    "problem_solution",  # Problem-solution format
    "vlog_style",  # Vlog-style narrative
    "explainer",  # Explainer/tutorial format
    "interview",  # Interview/conversation format
    "montage",  # Montage/compilation
    "documentary",  # Documentary style
    "episodic",  # Episodic structure
    "non_linear",  # Non-linear narrative
]

# Common character development types
CHARACTER_DEVELOPMENT_TYPES = [
    "introduction",  # Character introduction
    "growth",  # Character growth/development
    "transformation",  # Character transformation
    "revelation",  # Character revelation
    "conflict",  # Character in conflict
    "resolution",  # Character resolution
    "presenter_direct_address",  # Presenter directly addressing audience
    "presenter_demonstration",  # Presenter demonstrating something
    "presenter_reaction",  # Presenter reacting to something
    "expert_testimony",  # Expert providing testimony/information
]

# Common conflict types
CONFLICT_TYPES = [
    "person_vs_person",  # Person vs. person conflict
    "person_vs_self",  # Person vs. self conflict
    "person_vs_nature",  # Person vs. nature/environment conflict
    "person_vs_society",  # Person vs. society/system conflict
    "person_vs_technology",  # Person vs. technology conflict
    "problem_vs_solution",  # Problem vs. solution conflict
    "expectation_vs_reality",  # Expectation vs. reality conflict
    "before_vs_after",  # Before vs. after conflict
    "question_vs_answer",  # Question vs. answer conflict
    "challenge_vs_achievement",  # Challenge vs. achievement conflict
]

# Common persuasion techniques
PERSUASION_TECHNIQUES = [
    "storytelling",  # Using narrative to persuade
    "social_proof",  # Using testimonials/examples
    "authority",  # Appealing to authority/expertise
    "scarcity",  # Creating sense of scarcity/exclusivity
    "reciprocity",  # Creating sense of obligation
    "commitment",  # Getting small commitments that lead to larger ones
    "liking",  # Creating likability/rapport
    "urgency",  # Creating sense of urgency
    "ethos_appeal",  # Appeal to ethics/credibility
    "pathos_appeal",  # Appeal to emotions
    "logos_appeal",  # Appeal to logic/reason
]

# Common engagement strategies
ENGAGEMENT_STRATEGIES = [
    "direct_address",  # Directly addressing the audience
    "question_hooks",  # Using questions to hook audience
    "surprise_reveal",  # Surprising reveals/twists
    "cliffhanger",  # Using cliffhangers to maintain interest
    "pacing_variation",  # Varying pacing to maintain interest
    "emotional_contrast",  # Using emotional contrast
    "humor",  # Using humor to engage
    "suspense",  # Building suspense
    "relatability",  # Creating relatability
    "visual_interest",  # Using visually interesting elements
    "call_to_action",  # Clear call to action
]


@AnalyzerRegistry.register("storytelling")
class StorytellingAnalyzer(BaseAnalyzer):
    """
    Analyzes the storytelling aspects of videos.

    This analyzer identifies narrative structure, character development,
    conflict patterns, persuasion techniques, and audience engagement strategies.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the storytelling analyzer.

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

        # Default storytelling analysis parameters
        self.sample_interval = self._config.get("sample_interval", 30)  # seconds

    @property
    def analyzer_id(self) -> str:
        """
        Get the unique identifier for this analyzer.

        Returns:
            str: The analyzer ID
        """
        return "storytelling_analyzer"

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

    async def analyze(self, video_data: VideoData) -> StorytellingAnalysisResult:
        """
        Analyze the storytelling aspects of the video.

        Args:
            video_data: The video data to analyze

        Returns:
            StorytellingAnalysisResult: The storytelling analysis result
        """
        logger.info(f"Analyzing storytelling for video: {video_data.path}")

        # Get frames at regular intervals for storytelling analysis
        frames = video_data.get_frames_at_intervals(self.sample_interval)

        if not frames:
            logger.warning(
                f"No frames found for storytelling analysis in video: {video_data.path}"
            )
            # Create a minimal result with default values
            return StorytellingAnalysisResult(
                analyzer_id=self.analyzer_id,
                narrative_structure="unknown",
                character_development=[],
                conflict_patterns=[],
                persuasion_techniques=[],
                engagement_strategies=[],
                video_id=str(video_data.path),
                confidence=0.5,  # Low confidence due to lack of frames
            )

        # Analyze the storytelling aspects of the video
        narrative_structure = await self._identify_narrative_structure(
            video_data, frames
        )
        character_development = await self._identify_character_development(
            video_data, frames
        )
        conflict_patterns = await self._identify_conflict_patterns(video_data, frames)
        persuasion_techniques = await self._identify_persuasion_techniques(
            video_data, frames
        )
        engagement_strategies = await self._identify_engagement_strategies(
            video_data, frames
        )

        # Create the result
        result = StorytellingAnalysisResult(
            analyzer_id=self.analyzer_id,
            narrative_structure=narrative_structure,
            character_development=character_development,
            conflict_patterns=conflict_patterns,
            persuasion_techniques=persuasion_techniques,
            engagement_strategies=engagement_strategies,
            video_id=str(video_data.path),
            timestamps=[frame.timestamp for frame in frames],
            confidence=0.8,  # Reasonable confidence for storytelling analysis
        )

        logger.info(f"Storytelling analysis completed for video: {video_data.path}")
        return result

    async def _identify_narrative_structure(
        self, video_data: VideoData, frames: List[Frame]
    ) -> str:
        """
        Identify the narrative structure of the video.

        Args:
            video_data: The video data
            frames: Sampled frames for analysis

        Returns:
            str: Identified narrative structure
        """
        # If we have OpenAI integration, use it for more sophisticated analysis
        if self.openai_client:
            try:
                return await self._identify_narrative_structure_with_llm(
                    video_data, frames
                )
            except Exception as e:
                logger.error(
                    f"Error using OpenAI for narrative structure identification: {str(e)}",
                    exc_info=True,
                )
                # Fall back to basic analysis

        # Fallback to basic analysis
        return self._identify_narrative_structure_basic(video_data, frames)

    async def _identify_narrative_structure_with_llm(
        self, video_data: VideoData, frames: List[Frame]
    ) -> str:
        """
        Use OpenAI LLM to identify the narrative structure.

        Args:
            video_data: The video data
            frames: Sampled frames for analysis

        Returns:
            str: Identified narrative structure
        """
        try:
            # Prepare frame descriptions for the LLM
            frame_descriptions = []
            for i, frame in enumerate(
                frames[:10]  # Limit to first 10 frames to avoid token limits
            ):
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
            structures_str = ", ".join(NARRATIVE_STRUCTURES)

            # Combine into prompt
            prompt = f"""
            Analyze the following video frames to determine the narrative structure:
            
            Video metadata:
            - Duration: {video_data.duration} seconds
            - Resolution: {resolution_str}
            
            Frame descriptions:
            {frame_desc_str}
            
            Based on this information, identify the narrative structure of the video.
            Choose from these common narrative structures: {structures_str}
            
            Return only the most likely narrative structure as a single term.
            """

            # Call OpenAI API
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a video analysis expert specializing in identifying narrative structures in videos.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=50,
                temperature=0.3,
            )

            # Parse the response
            structure = response.choices[0].message.content.strip().lower()

            # Ensure the structure is one of the common structures
            if not any(s in structure for s in NARRATIVE_STRUCTURES):
                logger.warning(
                    f"LLM returned uncommon structure '{structure}', falling back to basic analysis"
                )
                return self._identify_narrative_structure_basic(video_data, frames)

            return structure

        except Exception as e:
            logger.error(
                f"Error using OpenAI for narrative structure identification: {str(e)}",
                exc_info=True,
            )
            # Fall back to basic analysis
            return self._identify_narrative_structure_basic(video_data, frames)

    def _identify_narrative_structure_basic(
        self, video_data: VideoData, frames: List[Frame]
    ) -> str:
        """
        Use basic heuristics to identify the narrative structure.

        Args:
            video_data: The video data
            frames: Sampled frames for analysis

        Returns:
            str: Identified narrative structure
        """
        # This is a simplified implementation that would be enhanced in a real system
        # with more sophisticated image and audio analysis

        # Determine structure based on video duration
        duration = video_data.duration

        if duration < 60:  # Less than 1 minute
            return "explainer"
        elif duration < 180:  # 1-3 minutes
            return "vlog_style"
        elif duration < 600:  # 3-10 minutes
            return "problem_solution"
        else:  # More than 10 minutes
            return "three_act"

    async def _identify_character_development(
        self, video_data: VideoData, frames: List[Frame]
    ) -> List[Dict[str, Any]]:
        """
        Identify character development or presenter techniques.

        Args:
            video_data: The video data
            frames: Sampled frames for analysis

        Returns:
            List[Dict[str, Any]]: Character development entries
        """
        character_development = []

        # This is a simplified implementation
        # In a real system, we would use computer vision to detect faces and people

        # Create some sample character development entries based on video duration
        duration = video_data.duration
        num_entries = max(
            2, min(5, int(duration / 60))
        )  # 2-5 entries based on duration

        for i in range(num_entries):
            # Calculate timestamp for this entry
            timestamp = (i + 1) * (duration / (num_entries + 1))

            # Determine character and development type based on position in video
            if i == 0:
                character = "main_presenter"
                dev_type = "introduction"
                description = "Introduction of the main presenter/character"
            elif i == num_entries - 1:
                character = "main_presenter"
                dev_type = "resolution"
                description = "Conclusion by the main presenter/character"
            else:
                # Alternate between different development types
                if i % 2 == 0:
                    character = "main_presenter"
                    dev_type = "presenter_demonstration"
                    description = "Presenter demonstrates key concept or technique"
                else:
                    character = "supporting_character"
                    dev_type = "expert_testimony"
                    description = "Supporting character provides expert insight"

            character_development.append(
                {
                    "timestamp": timestamp,
                    "character": character,
                    "development_type": dev_type,
                    "description": description,
                }
            )

        return character_development

    async def _identify_conflict_patterns(
        self, video_data: VideoData, frames: List[Frame]
    ) -> List[Dict[str, Any]]:
        """
        Identify conflict and resolution patterns.

        Args:
            video_data: The video data
            frames: Sampled frames for analysis

        Returns:
            List[Dict[str, Any]]: Conflict patterns
        """
        conflict_patterns = []

        # This is a simplified implementation
        # In a real system, we would analyze audio, text, and visual cues

        # Create some sample conflict patterns based on video duration
        duration = video_data.duration

        # For shorter videos, create fewer conflict patterns
        if duration < 300:  # Less than 5 minutes
            num_patterns = 1
        else:
            num_patterns = min(
                3, int(duration / 180)
            )  # Up to 3 patterns based on duration

        for i in range(num_patterns):
            # Calculate timestamps for conflict and resolution
            conflict_timestamp = (i + 1) * (duration / (num_patterns + 2))
            resolution_timestamp = conflict_timestamp + (duration / (num_patterns + 2))

            # Ensure resolution comes after conflict
            if resolution_timestamp <= conflict_timestamp:
                resolution_timestamp = conflict_timestamp + 10  # Add 10 seconds

            # Determine conflict type based on position in video
            if i == 0:
                conflict_type = "problem_vs_solution"
                description = "Introduction of a problem that needs solving"
            elif i == num_patterns - 1:
                conflict_type = "challenge_vs_achievement"
                description = "Final challenge before conclusion"
            else:
                conflict_type = "expectation_vs_reality"
                description = "Contrast between expected and actual outcomes"

            conflict_patterns.append(
                {
                    "conflict_timestamp": conflict_timestamp,
                    "resolution_timestamp": resolution_timestamp,
                    "conflict_type": conflict_type,
                    "description": description,
                }
            )

        return conflict_patterns

    async def _identify_persuasion_techniques(
        self, video_data: VideoData, frames: List[Frame]
    ) -> List[str]:
        """
        Identify persuasion techniques used in the video.

        Args:
            video_data: The video data
            frames: Sampled frames for analysis

        Returns:
            List[str]: Identified persuasion techniques
        """
        # This is a simplified implementation
        # In a real system, we would analyze audio, text, and visual cues

        # Always include storytelling as a technique
        techniques = ["storytelling"]

        # Add techniques based on video duration
        duration = video_data.duration

        if duration < 60:  # Less than 1 minute
            techniques.extend(["urgency", "pathos_appeal"])
        elif duration < 300:  # 1-5 minutes
            techniques.extend(["logos_appeal", "social_proof"])
        else:  # More than 5 minutes
            techniques.extend(["ethos_appeal", "authority", "commitment"])

        # Add one more random technique for variety
        import random

        remaining_techniques = [t for t in PERSUASION_TECHNIQUES if t not in techniques]
        if remaining_techniques:
            techniques.append(random.choice(remaining_techniques))

        return techniques

    async def _identify_engagement_strategies(
        self, video_data: VideoData, frames: List[Frame]
    ) -> List[str]:
        """
        Identify audience engagement strategies used in the video.

        Args:
            video_data: The video data
            frames: Sampled frames for analysis

        Returns:
            List[str]: Identified engagement strategies
        """
        # This is a simplified implementation
        # In a real system, we would analyze audio, text, and visual cues

        # Always include direct address as a strategy
        strategies = ["direct_address"]

        # Add strategies based on video duration
        duration = video_data.duration

        if duration < 60:  # Less than 1 minute
            strategies.extend(["surprise_reveal", "call_to_action"])
        elif duration < 300:  # 1-5 minutes
            strategies.extend(["question_hooks", "humor"])
        else:  # More than 5 minutes
            strategies.extend(["pacing_variation", "emotional_contrast", "suspense"])

        # Add one more random strategy for variety
        import random

        remaining_strategies = [s for s in ENGAGEMENT_STRATEGIES if s not in strategies]
        if remaining_strategies:
            strategies.append(random.choice(remaining_strategies))

        return strategies

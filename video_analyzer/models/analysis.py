"""
Data models for analysis results.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator, root_validator
import uuid


class AnalysisResult(BaseModel):
    """
    Base data model for analysis results.

    Attributes:
        analyzer_id: Unique identifier for the analyzer
        timestamp: Time when analysis was performed
        confidence: Confidence level of the analysis (0.0-1.0)
        data: Analysis data
        video_id: Identifier for the analyzed video
        timestamps: List of relevant timestamps in the video
    """

    analyzer_id: str = Field(..., description="Unique identifier for the analyzer")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Time when analysis was performed"
    )
    confidence: float = Field(
        default=1.0, description="Confidence level of the analysis (0.0-1.0)"
    )
    data: Dict[str, Any] = Field(default_factory=dict, description="Analysis data")
    video_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Identifier for the analyzed video",
    )
    timestamps: List[float] = Field(
        default_factory=list, description="List of relevant timestamps in the video"
    )

    @validator("confidence")
    def validate_confidence(cls, v):
        """Validate that confidence is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {v}")
        return v

    @validator("timestamps")
    def validate_timestamps(cls, v):
        """Validate that timestamps are non-negative."""
        for timestamp in v:
            if timestamp < 0:
                raise ValueError(f"Timestamps must be non-negative, got {timestamp}")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert the analysis result to a dictionary."""
        return self.dict()

    def get_summary(self) -> str:
        """Get a summary of the analysis result."""
        return f"Analysis by {self.analyzer_id} with confidence {self.confidence:.2f}"


class HookAnalysisResult(AnalysisResult):
    """
    Data model for hook analysis results.

    Attributes:
        hook_start_time: Start time of the hook section in seconds
        hook_end_time: End time of the hook section in seconds
        hook_techniques: Techniques used in the hook
        hook_effectiveness: Effectiveness score of the hook (0.0-1.0)
        key_moments: Key moments in the hook with timestamps
        recommendations: Recommendations for improving the hook
    """

    hook_start_time: float = Field(
        ..., description="Start time of the hook section in seconds"
    )
    hook_end_time: float = Field(
        ..., description="End time of the hook section in seconds"
    )
    hook_techniques: List[str] = Field(
        default_factory=list, description="Techniques used in the hook"
    )
    hook_effectiveness: float = Field(
        ..., description="Effectiveness score of the hook (0.0-1.0)"
    )
    key_moments: List[Dict[str, Any]] = Field(
        default_factory=list, description="Key moments in the hook"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations for improving the hook"
    )

    @validator("hook_start_time", "hook_end_time")
    def validate_times(cls, v):
        """Validate that times are non-negative."""
        if v < 0:
            raise ValueError(f"Time values must be non-negative, got {v}")
        return v

    @root_validator(skip_on_failure=True)
    def validate_hook_times(cls, values):
        """Validate that hook_end_time is after hook_start_time."""
        start = values.get("hook_start_time")
        end = values.get("hook_end_time")
        if start is not None and end is not None and start >= end:
            raise ValueError(
                f"Hook end time ({end}) must be after start time ({start})"
            )
        return values

    @validator("hook_effectiveness")
    def validate_effectiveness(cls, v):
        """Validate that effectiveness is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError(f"Effectiveness must be between 0 and 1, got {v}")
        return v

    @validator("key_moments")
    def validate_key_moments(cls, v):
        """Validate that key moments have required fields."""
        for moment in v:
            if "timestamp" not in moment:
                raise ValueError(f"Key moment must have a timestamp: {moment}")
            if "description" not in moment:
                raise ValueError(f"Key moment must have a description: {moment}")
        return v

    def get_hook_duration(self) -> float:
        """Get the duration of the hook in seconds."""
        return self.hook_end_time - self.hook_start_time


class ProgressionAnalysisResult(AnalysisResult):
    """
    Data model for progression analysis results.

    Attributes:
        sections: Video sections/segments with timestamps
        pacing_changes: Pacing changes throughout the video
        transitions: Transitions between topics or scenes
        narrative_flow_score: Score for narrative flow and coherence (0.0-1.0)
        retention_strategies: Identified retention strategies
    """

    sections: List[Dict[str, Any]] = Field(
        default_factory=list, description="Video sections/segments"
    )
    pacing_changes: List[Dict[str, Any]] = Field(
        default_factory=list, description="Pacing changes throughout the video"
    )
    transitions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Transitions between topics or scenes"
    )
    narrative_flow_score: float = Field(
        ..., description="Score for narrative flow and coherence (0.0-1.0)"
    )
    retention_strategies: List[str] = Field(
        default_factory=list, description="Identified retention strategies"
    )

    @validator("narrative_flow_score")
    def validate_flow_score(cls, v):
        """Validate that narrative flow score is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError(f"Narrative flow score must be between 0 and 1, got {v}")
        return v

    @validator("sections")
    def validate_sections(cls, v):
        """Validate that sections have required fields."""
        for section in v:
            if "start_time" not in section:
                raise ValueError(f"Section must have a start_time: {section}")
            if "end_time" not in section:
                raise ValueError(f"Section must have an end_time: {section}")
            if "title" not in section:
                raise ValueError(f"Section must have a title: {section}")

            # Validate that end_time is after start_time
            if section["end_time"] <= section["start_time"]:
                raise ValueError(
                    f"Section end time ({section['end_time']}) must be after start time ({section['start_time']})"
                )
        return v

    @validator("transitions")
    def validate_transitions(cls, v):
        """Validate that transitions have required fields."""
        for transition in v:
            if "timestamp" not in transition:
                raise ValueError(f"Transition must have a timestamp: {transition}")
            if "type" not in transition:
                raise ValueError(f"Transition must have a type: {transition}")
        return v

    def get_section_at_time(self, timestamp: float) -> Optional[Dict[str, Any]]:
        """Get the section that contains the specified timestamp."""
        for section in self.sections:
            if section["start_time"] <= timestamp <= section["end_time"]:
                return section
        return None


class VisualAnalysisResult(AnalysisResult):
    """
    Data model for visual analysis results.

    Attributes:
        lighting_quality: Lighting quality score (0.0-1.0)
        color_schemes: Identified color schemes with timestamps
        camera_movements: Detected camera movements with timestamps
        visual_effects: Identified visual effects with timestamps
        visual_recommendations: Recommendations for visual improvements
    """

    lighting_quality: float = Field(..., description="Lighting quality score (0.0-1.0)")
    color_schemes: List[Dict[str, Any]] = Field(
        default_factory=list, description="Identified color schemes"
    )
    camera_movements: List[Dict[str, Any]] = Field(
        default_factory=list, description="Detected camera movements"
    )
    visual_effects: List[Dict[str, Any]] = Field(
        default_factory=list, description="Identified visual effects"
    )
    visual_recommendations: List[str] = Field(
        default_factory=list, description="Recommendations for visual improvements"
    )

    @validator("lighting_quality")
    def validate_lighting_quality(cls, v):
        """Validate that lighting quality is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError(f"Lighting quality must be between 0 and 1, got {v}")
        return v

    @validator("color_schemes")
    def validate_color_schemes(cls, v):
        """Validate that color schemes have required fields."""
        for scheme in v:
            if "timestamp" not in scheme:
                raise ValueError(f"Color scheme must have a timestamp: {scheme}")
            if "colors" not in scheme:
                raise ValueError(f"Color scheme must have colors: {scheme}")
            if "mood" not in scheme:
                raise ValueError(f"Color scheme must have a mood: {scheme}")
        return v

    @validator("camera_movements")
    def validate_camera_movements(cls, v):
        """Validate that camera movements have required fields."""
        for movement in v:
            if "timestamp" not in movement:
                raise ValueError(f"Camera movement must have a timestamp: {movement}")
            if "type" not in movement:
                raise ValueError(f"Camera movement must have a type: {movement}")
            if "duration" not in movement:
                raise ValueError(f"Camera movement must have a duration: {movement}")
        return v

    @validator("visual_effects")
    def validate_visual_effects(cls, v):
        """Validate that visual effects have required fields."""
        for effect in v:
            if "timestamp" not in effect:
                raise ValueError(f"Visual effect must have a timestamp: {effect}")
            if "type" not in effect:
                raise ValueError(f"Visual effect must have a type: {effect}")
            if "purpose" not in effect:
                raise ValueError(f"Visual effect must have a purpose: {effect}")
        return v

    def get_effects_at_time(
        self, timestamp: float, margin: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Get visual effects at or near the specified timestamp.

        Args:
            timestamp: Time in seconds
            margin: Time margin in seconds to consider

        Returns:
            List of visual effects at or near the timestamp
        """
        return [
            effect
            for effect in self.visual_effects
            if abs(effect["timestamp"] - timestamp) <= margin
        ]


class AudioAnalysisResult(AnalysisResult):
    """
    Data model for audio analysis results.

    Attributes:
        sound_quality: Sound quality score (0.0-1.0)
        speech_analysis: Analysis of speech patterns (pacing, tone, delivery)
        background_music: Identified background music with timestamps
        sound_effects: Detected sound effects with timestamps
        transcription: Speech transcription
    """

    sound_quality: float = Field(..., description="Sound quality score (0.0-1.0)")
    speech_analysis: Dict[str, Any] = Field(
        default_factory=dict, description="Analysis of speech patterns"
    )
    background_music: List[Dict[str, Any]] = Field(
        default_factory=list, description="Identified background music"
    )
    sound_effects: List[Dict[str, Any]] = Field(
        default_factory=list, description="Detected sound effects"
    )
    transcription: str = Field("", description="Speech transcription")

    @validator("sound_quality")
    def validate_sound_quality(cls, v):
        """Validate that sound quality is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError(f"Sound quality must be between 0 and 1, got {v}")
        return v

    @validator("speech_analysis")
    def validate_speech_analysis(cls, v):
        """Validate that speech analysis has required fields."""
        required_fields = ["pacing", "tone", "clarity"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Speech analysis must include {field}")
        return v

    @validator("background_music")
    def validate_background_music(cls, v):
        """Validate that background music entries have required fields."""
        for music in v:
            if "start_time" not in music:
                raise ValueError(f"Background music must have a start_time: {music}")
            if "end_time" not in music:
                raise ValueError(f"Background music must have an end_time: {music}")
            if "mood" not in music:
                raise ValueError(f"Background music must have a mood: {music}")

            # Validate that end_time is after start_time
            if music["end_time"] <= music["start_time"]:
                raise ValueError(
                    f"Music end time ({music['end_time']}) must be after start time ({music['start_time']})"
                )
        return v

    @validator("sound_effects")
    def validate_sound_effects(cls, v):
        """Validate that sound effects have required fields."""
        for effect in v:
            if "timestamp" not in effect:
                raise ValueError(f"Sound effect must have a timestamp: {effect}")
            if "type" not in effect:
                raise ValueError(f"Sound effect must have a type: {effect}")
            if "purpose" not in effect:
                raise ValueError(f"Sound effect must have a purpose: {effect}")
        return v

    def get_music_at_time(self, timestamp: float) -> List[Dict[str, Any]]:
        """
        Get background music playing at the specified timestamp.

        Args:
            timestamp: Time in seconds

        Returns:
            List of background music entries active at the timestamp
        """
        return [
            music
            for music in self.background_music
            if music["start_time"] <= timestamp <= music["end_time"]
        ]


class ObjectDetectionResult(AnalysisResult):
    """
    Data model for object detection results.

    Attributes:
        objects: Detected objects with timestamps and positions
        faces: Detected faces and expressions with timestamps
        brands: Identified brand logos and products with timestamps
        screen_time_analysis: Screen time analysis of objects
        brand_integration_score: Brand integration score (0.0-1.0)
    """

    objects: List[Dict[str, Any]] = Field(
        default_factory=list, description="Detected objects"
    )
    faces: List[Dict[str, Any]] = Field(
        default_factory=list, description="Detected faces and expressions"
    )
    brands: List[Dict[str, Any]] = Field(
        default_factory=list, description="Identified brand logos and products"
    )
    screen_time_analysis: Dict[str, Any] = Field(
        default_factory=dict, description="Screen time analysis of objects"
    )
    brand_integration_score: float = Field(
        0.0, description="Brand integration score (0.0-1.0)"
    )

    @validator("brand_integration_score")
    def validate_brand_score(cls, v):
        """Validate that brand integration score is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError(
                f"Brand integration score must be between 0 and 1, got {v}"
            )
        return v

    @validator("objects")
    def validate_objects(cls, v):
        """Validate that objects have required fields."""
        for obj in v:
            if "timestamp" not in obj:
                raise ValueError(f"Object must have a timestamp: {obj}")
            if "label" not in obj:
                raise ValueError(f"Object must have a label: {obj}")
            if "confidence" not in obj:
                raise ValueError(f"Object must have a confidence: {obj}")
            if "bounding_box" not in obj:
                raise ValueError(f"Object must have a bounding_box: {obj}")
        return v

    @validator("faces")
    def validate_faces(cls, v):
        """Validate that faces have required fields."""
        for face in v:
            if "timestamp" not in face:
                raise ValueError(f"Face must have a timestamp: {face}")
            if "bounding_box" not in face:
                raise ValueError(f"Face must have a bounding_box: {face}")
            if "expression" not in face:
                raise ValueError(f"Face must have an expression: {face}")
        return v

    @validator("brands")
    def validate_brands(cls, v):
        """Validate that brands have required fields."""
        for brand in v:
            if "timestamp" not in brand:
                raise ValueError(f"Brand must have a timestamp: {brand}")
            if "name" not in brand:
                raise ValueError(f"Brand must have a name: {brand}")
            if "bounding_box" not in brand:
                raise ValueError(f"Brand must have a bounding_box: {brand}")
        return v

    def get_objects_at_time(
        self, timestamp: float, margin: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Get objects detected at or near the specified timestamp.

        Args:
            timestamp: Time in seconds
            margin: Time margin in seconds to consider

        Returns:
            List of objects at or near the timestamp
        """
        return [
            obj for obj in self.objects if abs(obj["timestamp"] - timestamp) <= margin
        ]

    def get_screen_time_for_object(self, object_label: str) -> float:
        """
        Get the total screen time for a specific object type.

        Args:
            object_label: The object label to look for

        Returns:
            Total screen time in seconds
        """
        if (
            "by_label" in self.screen_time_analysis
            and object_label in self.screen_time_analysis["by_label"]
        ):
            return self.screen_time_analysis["by_label"][object_label]
        return 0.0


class EmotionAnalysisResult(AnalysisResult):
    """
    Data model for emotion analysis results.

    Attributes:
        overall_mood: Overall mood/tone of the video
        emotional_shifts: Emotional shifts throughout the video with timestamps
        emotional_elements: Visual and audio elements contributing to emotions
        emotion_techniques: Techniques used to elicit specific emotions
        emotional_journey: Emotional journey map of the video with timestamps
    """

    overall_mood: str = Field(..., description="Overall mood/tone of the video")
    emotional_shifts: List[Dict[str, Any]] = Field(
        default_factory=list, description="Emotional shifts throughout the video"
    )
    emotional_elements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Visual and audio elements contributing to emotions",
    )
    emotion_techniques: List[str] = Field(
        default_factory=list, description="Techniques used to elicit specific emotions"
    )
    emotional_journey: List[Dict[str, Any]] = Field(
        default_factory=list, description="Emotional journey map of the video"
    )

    @validator("emotional_shifts")
    def validate_emotional_shifts(cls, v):
        """Validate that emotional shifts have required fields."""
        for shift in v:
            if "timestamp" not in shift:
                raise ValueError(f"Emotional shift must have a timestamp: {shift}")
            if "from_emotion" not in shift:
                raise ValueError(f"Emotional shift must have a from_emotion: {shift}")
            if "to_emotion" not in shift:
                raise ValueError(f"Emotional shift must have a to_emotion: {shift}")
            if "trigger" not in shift:
                raise ValueError(f"Emotional shift must have a trigger: {shift}")
        return v

    @validator("emotional_elements")
    def validate_emotional_elements(cls, v):
        """Validate that emotional elements have required categories."""
        required_categories = ["visual", "audio"]
        for category in required_categories:
            if category not in v:
                raise ValueError(f"Emotional elements must include {category} category")
        return v

    @validator("emotional_journey")
    def validate_emotional_journey(cls, v):
        """Validate that emotional journey entries have required fields."""
        for entry in v:
            if "timestamp" not in entry:
                raise ValueError(
                    f"Emotional journey entry must have a timestamp: {entry}"
                )
            if "emotion" not in entry:
                raise ValueError(
                    f"Emotional journey entry must have an emotion: {entry}"
                )
            if "intensity" not in entry:
                raise ValueError(
                    f"Emotional journey entry must have an intensity: {entry}"
                )

            # Validate intensity is between 0 and 1
            if not 0 <= entry["intensity"] <= 1:
                raise ValueError(
                    f"Emotional intensity must be between 0 and 1, got {entry['intensity']}"
                )
        return v

    def get_emotion_at_time(
        self, timestamp: float, margin: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """
        Get the emotional state at or near the specified timestamp.

        Args:
            timestamp: Time in seconds
            margin: Time margin in seconds to consider

        Returns:
            Emotional journey entry at or near the timestamp, or None if not found
        """
        # Sort by how close the timestamp is to the target
        matching_entries = sorted(
            [
                entry
                for entry in self.emotional_journey
                if abs(entry["timestamp"] - timestamp) <= margin
            ],
            key=lambda e: abs(e["timestamp"] - timestamp),
        )

        return matching_entries[0] if matching_entries else None


class StorytellingAnalysisResult(AnalysisResult):
    """
    Data model for storytelling analysis results.

    Attributes:
        narrative_structure: Identified narrative structure
        character_development: Character development or presenter techniques with timestamps
        conflict_patterns: Conflict and resolution patterns with timestamps
        persuasion_techniques: Identified persuasion techniques
        engagement_strategies: Audience engagement strategies
    """

    narrative_structure: str = Field(..., description="Identified narrative structure")
    character_development: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Character development or presenter techniques",
    )
    conflict_patterns: List[Dict[str, Any]] = Field(
        default_factory=list, description="Conflict and resolution patterns"
    )
    persuasion_techniques: List[str] = Field(
        default_factory=list, description="Identified persuasion techniques"
    )
    engagement_strategies: List[str] = Field(
        default_factory=list, description="Audience engagement strategies"
    )

    @validator("narrative_structure")
    def validate_narrative_structure(cls, v):
        """Validate that narrative structure is not empty."""
        if not v:
            raise ValueError("Narrative structure cannot be empty")
        return v

    @validator("character_development")
    def validate_character_development(cls, v):
        """Validate that character development entries have required fields."""
        for entry in v:
            if "timestamp" not in entry:
                raise ValueError(
                    f"Character development entry must have a timestamp: {entry}"
                )
            if "character" not in entry:
                raise ValueError(
                    f"Character development entry must have a character: {entry}"
                )
            if "development_type" not in entry:
                raise ValueError(
                    f"Character development entry must have a development_type: {entry}"
                )
        return v

    @validator("conflict_patterns")
    def validate_conflict_patterns(cls, v):
        """Validate that conflict patterns have required fields."""
        for pattern in v:
            if "conflict_timestamp" not in pattern:
                raise ValueError(
                    f"Conflict pattern must have a conflict_timestamp: {pattern}"
                )
            if "resolution_timestamp" not in pattern:
                raise ValueError(
                    f"Conflict pattern must have a resolution_timestamp: {pattern}"
                )
            if "conflict_type" not in pattern:
                raise ValueError(
                    f"Conflict pattern must have a conflict_type: {pattern}"
                )

            # Validate that resolution_timestamp is after conflict_timestamp
            if pattern["resolution_timestamp"] <= pattern["conflict_timestamp"]:
                raise ValueError(
                    f"Resolution timestamp ({pattern['resolution_timestamp']}) must be after conflict timestamp ({pattern['conflict_timestamp']})"
                )
        return v

    def get_narrative_elements_at_time(
        self, timestamp: float, margin: float = 2.0
    ) -> Dict[str, Any]:
        """
        Get narrative elements at or near the specified timestamp.

        Args:
            timestamp: Time in seconds
            margin: Time margin in seconds to consider

        Returns:
            Dictionary with character development and conflict patterns at the timestamp
        """
        character_elements = [
            entry
            for entry in self.character_development
            if abs(entry["timestamp"] - timestamp) <= margin
        ]

        conflict_elements = [
            pattern
            for pattern in self.conflict_patterns
            if abs(pattern["conflict_timestamp"] - timestamp) <= margin
            or abs(pattern["resolution_timestamp"] - timestamp) <= margin
        ]

        return {
            "character_development": character_elements,
            "conflict_patterns": conflict_elements,
        }


class Report(BaseModel):
    """
    Data model for the final report.

    Attributes:
        video_id: Unique identifier for the video
        analysis_timestamp: Time when analysis was performed
        analysis_duration: Duration of the analysis in seconds
        summary: Summary of the analysis
        sections: Analysis sections organized by category
        recommendations: Overall recommendations
        visual_examples: Visual examples extracted from the video with timestamps
    """

    video_id: str = Field(..., description="Unique identifier for the video")
    analysis_timestamp: datetime = Field(
        default_factory=datetime.now, description="Time when analysis was performed"
    )
    analysis_duration: float = Field(
        ..., description="Duration of the analysis in seconds"
    )
    summary: str = Field(..., description="Summary of the analysis")
    sections: Dict[str, Any] = Field(
        default_factory=dict, description="Analysis sections"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Overall recommendations"
    )
    visual_examples: List[Dict[str, Any]] = Field(
        default_factory=list, description="Visual examples extracted from the video"
    )

    @validator("analysis_duration")
    def validate_analysis_duration(cls, v):
        """Validate that analysis duration is positive."""
        if v <= 0:
            raise ValueError(f"Analysis duration must be positive, got {v}")
        return v

    @validator("summary")
    def validate_summary(cls, v):
        """Validate that summary is not empty."""
        if not v:
            raise ValueError("Summary cannot be empty")
        return v

    @validator("sections")
    def validate_sections(cls, v):
        """Validate that sections contain required categories."""
        required_sections = [
            "hook",
            "progression",
            "visual",
            "audio",
            "objects",
            "emotion",
            "storytelling",
        ]

        for section in required_sections:
            if section not in v:
                raise ValueError(f"Report sections must include '{section}' category")
        return v

    @validator("visual_examples")
    def validate_visual_examples(cls, v):
        """Validate that visual examples have required fields."""
        for example in v:
            if "timestamp" not in example:
                raise ValueError(f"Visual example must have a timestamp: {example}")
            if "description" not in example:
                raise ValueError(f"Visual example must have a description: {example}")
            if "category" not in example:
                raise ValueError(f"Visual example must have a category: {example}")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert the report to a dictionary."""
        return self.dict()

    def get_section(self, section_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific section of the report.

        Args:
            section_name: Name of the section to retrieve

        Returns:
            The section data or None if not found
        """
        return self.sections.get(section_name)

    def get_examples_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get visual examples for a specific category.

        Args:
            category: Category name (e.g., 'hook', 'visual', 'emotion')

        Returns:
            List of visual examples for the category
        """
        return [
            example
            for example in self.visual_examples
            if example["category"] == category
        ]

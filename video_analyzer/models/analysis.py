"""
Data models for analysis results.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class AnalysisResult(BaseModel):
    """
    Base data model for analysis results.
    """

    analyzer_id: str = Field(..., description="Unique identifier for the analyzer")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Time when analysis was performed"
    )
    confidence: float = Field(
        default=1.0, description="Confidence level of the analysis (0.0-1.0)"
    )
    data: Dict[str, Any] = Field(default_factory=dict, description="Analysis data")


class HookAnalysisResult(AnalysisResult):
    """
    Data model for hook analysis results.
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


class ProgressionAnalysisResult(AnalysisResult):
    """
    Data model for progression analysis results.
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


class VisualAnalysisResult(AnalysisResult):
    """
    Data model for visual analysis results.
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


class AudioAnalysisResult(AnalysisResult):
    """
    Data model for audio analysis results.
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


class ObjectDetectionResult(AnalysisResult):
    """
    Data model for object detection results.
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


class EmotionAnalysisResult(AnalysisResult):
    """
    Data model for emotion analysis results.
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


class StorytellingAnalysisResult(AnalysisResult):
    """
    Data model for storytelling analysis results.
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


class Report(BaseModel):
    """
    Data model for the final report.
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

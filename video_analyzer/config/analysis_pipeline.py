"""
Configuration for the AnalysisPipeline.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator


class AnalysisPipelineConfig(BaseModel):
    """
    Configuration for the AnalysisPipeline.

    Attributes:
        parallel_analyzers: Whether to run analyzers in parallel
        timeout_seconds: Timeout for each analyzer in seconds
        max_retries: Maximum number of retries for failed analyzers
        analyzer_weights: Weights for each analyzer type (used for progress calculation)
        enabled_analyzers: List of enabled analyzer types (None for all)
    """

    parallel_analyzers: bool = Field(
        default=True,
        description="Whether to run analyzers in parallel",
    )
    timeout_seconds: int = Field(
        default=300,
        description="Timeout for each analyzer in seconds",
    )
    max_retries: int = Field(
        default=2,
        description="Maximum number of retries for failed analyzers",
    )
    analyzer_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "hook": 1.0,
            "progression": 1.0,
            "visual": 1.0,
            "audio": 1.0,
            "object": 1.0,
            "emotion": 1.0,
            "storytelling": 1.0,
        },
        description="Weights for each analyzer type (used for progress calculation)",
    )
    enabled_analyzers: Optional[List[str]] = Field(
        default=None,
        description="List of enabled analyzer types (None for all)",
    )

    @validator("timeout_seconds")
    def validate_timeout(cls, v):
        """Validate that timeout is positive."""
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v

    @validator("max_retries")
    def validate_max_retries(cls, v):
        """Validate that max_retries is non-negative."""
        if v < 0:
            raise ValueError("Max retries must be non-negative")
        return v

    @validator("analyzer_weights")
    def validate_analyzer_weights(cls, v):
        """Validate that analyzer weights are positive."""
        for analyzer, weight in v.items():
            if weight <= 0:
                raise ValueError(f"Weight for analyzer {analyzer} must be positive")
        return v

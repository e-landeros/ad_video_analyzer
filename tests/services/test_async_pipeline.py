"""
Tests for the asynchronous processing capabilities of the analysis pipeline.
"""

import asyncio
import pytest
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

from video_analyzer.config.analysis_pipeline import AnalysisPipelineConfig
from video_analyzer.services.pipeline import AnalysisPipeline
from video_analyzer.analyzers.base import (
    BaseAnalyzer,
    AnalysisProgressCallback,
    AnalysisCancellationToken,
    CancellationError,
)
from video_analyzer.models.video import VideoData, Frame
from video_analyzer.models.analysis import AnalysisResult


class MockAnalyzer(BaseAnalyzer):
    """Mock analyzer for testing."""

    def __init__(
        self, analyzer_id, delay=0.1, should_fail=False, supports_progress=True
    ):
        super().__init__()
        self._analyzer_id = analyzer_id
        self.delay = delay
        self.should_fail = should_fail
        self._supports_progress = supports_progress

    async def analyze(self, video_data):
        """Mock analyze method."""
        # Simulate work
        await asyncio.sleep(self.delay)

        if self.should_fail:
            raise ValueError(f"Mock analyzer {self.analyzer_id} failed")

        return AnalysisResult(
            analyzer_id=self.analyzer_id,
            confidence=0.9,
            data={"mock": "data"},
            video_id=str(video_data.path),
        )

    @property
    def analyzer_id(self):
        return self._analyzer_id

    @property
    def analyzer_category(self):
        return "mock"

    @property
    def supports_progress(self):
        return self._supports_progress


@pytest.fixture
def mock_video_data():
    """Create mock video data for testing."""
    # Create a dummy frame
    dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
    frames = [Frame(image=dummy_image, timestamp=i, index=i) for i in range(5)]

    return VideoData(
        path=Path("test_video.mp4"),
        frames=frames,
        duration=10.0,
        fps=30.0,
        resolution=(1920, 1080),
    )


@pytest.fixture
def pipeline_config():
    """Create a pipeline configuration for testing."""
    return AnalysisPipelineConfig(
        parallel_analyzers=True,
        timeout_seconds=5,
        max_retries=1,
        analyzer_weights={"mock": 1.0},
        progress_update_interval=0.1,
        cancellation_check_interval=0.1,
    )


@pytest.mark.asyncio
async def test_parallel_analysis(mock_video_data, pipeline_config):
    """Test running analyzers in parallel."""
    # Create a pipeline
    pipeline = AnalysisPipeline(pipeline_config)

    # Register some mock analyzers
    for i in range(3):
        pipeline.register_analyzer(MockAnalyzer(f"mock_analyzer_{i}", delay=0.2))

    # Set up progress tracking
    progress_updates = []

    def on_progress(analyzer_id, progress, metadata):
        progress_updates.append((analyzer_id, progress, metadata))

    pipeline.set_progress_callback(on_progress)

    # Run the analysis
    start_time = time.time()
    results = await pipeline.run_analysis(mock_video_data)
    elapsed_time = time.time() - start_time

    # Check that all analyzers ran
    assert len(results) == 3

    # Check that they ran in parallel (elapsed time should be close to the max delay)
    assert elapsed_time < 0.6  # 0.2s delay + some overhead

    # Check that we got progress updates
    assert len(progress_updates) > 0

    # Check that we got a final progress update with progress=1.0
    assert any(
        update[0] == "pipeline" and update[1] == 1.0 for update in progress_updates
    )


@pytest.mark.asyncio
async def test_sequential_analysis(mock_video_data, pipeline_config):
    """Test running analyzers sequentially."""
    # Create a pipeline with sequential execution
    pipeline_config.parallel_analyzers = False
    pipeline = AnalysisPipeline(pipeline_config)

    # Register some mock analyzers
    for i in range(3):
        pipeline.register_analyzer(MockAnalyzer(f"mock_analyzer_{i}", delay=0.2))

    # Run the analysis
    start_time = time.time()
    results = await pipeline.run_analysis(mock_video_data)
    elapsed_time = time.time() - start_time

    # Check that all analyzers ran
    assert len(results) == 3

    # Check that they ran sequentially (elapsed time should be sum of delays)
    assert elapsed_time >= 0.6  # 3 * 0.2s delay


@pytest.mark.asyncio
async def test_cancellation(mock_video_data, pipeline_config):
    """Test cancelling the analysis."""
    # Create a pipeline
    pipeline = AnalysisPipeline(pipeline_config)

    # Register some mock analyzers with longer delays
    for i in range(3):
        pipeline.register_analyzer(MockAnalyzer(f"mock_analyzer_{i}", delay=1.0))

    # Create a cancellation token
    cancellation_token = pipeline.create_cancellation_token()

    # Set up a task to cancel the analysis after a short delay
    async def cancel_after_delay():
        await asyncio.sleep(0.5)
        cancellation_token.cancel()

    # Run the analysis and cancellation concurrently
    cancel_task = asyncio.create_task(cancel_after_delay())

    # The analysis should be cancelled
    with pytest.raises(CancellationError):
        await pipeline.run_analysis(mock_video_data)

    # Clean up
    await cancel_task


@pytest.mark.asyncio
async def test_analyzer_failure(mock_video_data, pipeline_config):
    """Test handling analyzer failures."""
    # Create a pipeline that doesn't fail fast
    pipeline_config.fail_fast = False
    pipeline = AnalysisPipeline(pipeline_config)

    # Register some mock analyzers, one of which will fail
    pipeline.register_analyzer(MockAnalyzer("mock_analyzer_1", delay=0.1))
    pipeline.register_analyzer(
        MockAnalyzer("mock_analyzer_2", delay=0.1, should_fail=True)
    )
    pipeline.register_analyzer(MockAnalyzer("mock_analyzer_3", delay=0.1))

    # Run the analysis
    results = await pipeline.run_analysis(mock_video_data)

    # Check that we got results from the non-failing analyzers
    assert len(results) == 2
    assert "mock_analyzer_1" in results
    assert "mock_analyzer_3" in results
    assert "mock_analyzer_2" not in results


@pytest.mark.asyncio
async def test_fail_fast(mock_video_data, pipeline_config):
    """Test fail-fast behavior."""
    # Create a pipeline that fails fast
    pipeline_config.fail_fast = True
    pipeline = AnalysisPipeline(pipeline_config)

    # Register some mock analyzers, one of which will fail
    pipeline.register_analyzer(MockAnalyzer("mock_analyzer_1", delay=0.1))
    pipeline.register_analyzer(
        MockAnalyzer("mock_analyzer_2", delay=0.1, should_fail=True)
    )
    pipeline.register_analyzer(MockAnalyzer("mock_analyzer_3", delay=0.1))

    # The analysis should fail
    with pytest.raises(Exception):
        await pipeline.run_analysis(mock_video_data)


@pytest.mark.asyncio
async def test_progress_tracking(mock_video_data, pipeline_config):
    """Test detailed progress tracking."""
    # Create a pipeline
    pipeline = AnalysisPipeline(pipeline_config)

    # Register some mock analyzers
    pipeline.register_analyzer(MockAnalyzer("mock_analyzer_1", delay=0.2))
    pipeline.register_analyzer(MockAnalyzer("mock_analyzer_2", delay=0.3))

    # Set up progress tracking
    progress_by_analyzer = {}

    def on_progress(analyzer_id, progress, metadata):
        if analyzer_id not in progress_by_analyzer:
            progress_by_analyzer[analyzer_id] = []
        progress_by_analyzer[analyzer_id].append((progress, metadata))

    pipeline.set_progress_callback(on_progress)

    # Run the analysis
    await pipeline.run_analysis(mock_video_data)

    # Check that we got progress updates for each analyzer
    assert "pipeline" in progress_by_analyzer
    assert "mock_analyzer_1" in progress_by_analyzer
    assert "mock_analyzer_2" in progress_by_analyzer

    # Check that each analyzer has at least initial (0.0) and final (1.0) progress
    for analyzer_id, updates in progress_by_analyzer.items():
        progress_values = [update[0] for update in updates]
        assert 0.0 in progress_values
        assert 1.0 in progress_values

    # Check that the pipeline progress reached 1.0
    pipeline_progress = [update[0] for update in progress_by_analyzer["pipeline"]]
    assert pipeline_progress[-1] == 1.0

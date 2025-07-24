"""
End-to-end integration tests for the video analyzer.

These tests verify the complete analysis workflow from video input to report generation.
"""

import os
import pytest
import asyncio
import tempfile
import json
import time
from pathlib import Path
from typing import Dict, Any, List

from video_analyzer.api.cli import app
from video_analyzer.services.analysis_manager import AnalysisManager
from video_analyzer.analyzers.base import AnalyzerRegistry
from video_analyzer.models.analysis import AnalysisResult, Report
from video_analyzer.utils.errors import VideoProcessingError
from typer.testing import CliRunner


@pytest.fixture
def runner():
    """Fixture for creating a CLI runner."""
    return CliRunner()


@pytest.fixture
def test_video_path():
    """
    Create a test video file for integration tests.

    This fixture creates a small test video file that can be used for integration tests.
    In a real-world scenario, you would have a collection of test videos with different
    characteristics.
    """
    # For actual testing, we need a real video file
    # This is a placeholder that points to a test video that should be created
    test_video_dir = Path("tests/test_data")
    test_video_dir.mkdir(exist_ok=True)

    test_video_path = test_video_dir / "test_video.mp4"

    # Check if the test video already exists
    if not test_video_path.exists():
        pytest.skip(
            f"Test video not found at {test_video_path}. Create a test video to run this test."
        )

    return test_video_path


@pytest.fixture
def create_test_videos():
    """
    Create test videos for different scenarios.

    This fixture creates a set of test videos with different characteristics
    that can be used for testing different aspects of the video analyzer.
    """
    # In a real implementation, this would create actual test videos
    # For now, we'll just create a directory structure
    test_video_dir = Path("tests/test_data")
    test_video_dir.mkdir(exist_ok=True)

    # Return the directory where test videos are stored
    return test_video_dir


class TestEndToEndAnalysis:
    """Test the end-to-end analysis workflow."""

    @pytest.mark.asyncio
    async def test_complete_analysis_workflow(self, test_video_path):
        """Test the complete analysis workflow from video input to report generation."""
        if not test_video_path.exists():
            pytest.skip(f"Test video not found at {test_video_path}")

        # Create an analysis manager
        manager = AnalysisManager()

        # Register all available analyzers
        for analyzer_type in AnalyzerRegistry.get_available_types():
            manager.register_analyzer(analyzer_type)

        # Set up progress tracking
        progress_updates = []

        def on_progress(
            analyzer_id: str, progress: float, metadata: Dict[str, Any]
        ) -> None:
            progress_updates.append((analyzer_id, progress, metadata))

        manager.set_progress_callback(on_progress)

        # Create a cancellation token (but don't cancel)
        cancellation_token = manager.create_cancellation_token()

        # Run the analysis
        start_time = time.time()
        results = await manager.analyze_video(test_video_path)
        end_time = time.time()

        # Verify results
        assert results, "Analysis should return results"
        assert isinstance(results, dict), "Results should be a dictionary"
        assert len(results) > 0, "There should be at least one analyzer result"

        # Check that each result is an AnalysisResult
        for analyzer_id, result in results.items():
            assert isinstance(result, AnalysisResult), (
                f"Result for {analyzer_id} should be an AnalysisResult"
            )
            assert result.analyzer_id == analyzer_id, (
                f"Result analyzer_id should match {analyzer_id}"
            )
            assert result.confidence >= 0 and result.confidence <= 1, (
                f"Confidence should be between 0 and 1"
            )

        # Generate a report
        report = manager.generate_report(results, str(test_video_path))

        # Verify report
        assert isinstance(report, Report), "Report should be a Report object"
        assert report.video_id == str(test_video_path), (
            "Report video_id should match the input video"
        )
        assert report.analysis_duration > 0, "Analysis duration should be positive"
        assert report.sections, "Report should have sections"

        # Verify progress tracking
        assert len(progress_updates) > 0, "There should be progress updates"

        # Verify execution time
        assert end_time - start_time > 0, "Analysis should take some time"
        assert manager.execution_time > 0, "Manager should track execution time"

        # Print some information about the analysis
        print(f"Analysis completed in {manager.execution_time:.2f} seconds")
        print(f"Number of analyzers: {len(results)}")
        print(f"Number of progress updates: {len(progress_updates)}")

    @pytest.mark.asyncio
    async def test_analysis_with_specific_analyzers(self, test_video_path):
        """Test analysis with specific analyzers."""
        if not test_video_path.exists():
            pytest.skip(f"Test video not found at {test_video_path}")

        # Create an analysis manager
        manager = AnalysisManager()

        # Register only specific analyzers
        specific_analyzers = ["hook", "progression"]
        for analyzer_type in specific_analyzers:
            manager.register_analyzer(analyzer_type)

        # Run the analysis
        results = await manager.analyze_video(test_video_path)

        # Verify results
        assert results, "Analysis should return results"
        assert len(results) == len(specific_analyzers), (
            f"There should be {len(specific_analyzers)} analyzer results"
        )

        # Check that each specified analyzer has a result
        for analyzer_type in specific_analyzers:
            assert any(
                analyzer_type in analyzer_id for analyzer_id in results.keys()
            ), f"Results should include {analyzer_type} analyzer"

    @pytest.mark.asyncio
    async def test_analysis_with_invalid_video(self):
        """Test analysis with an invalid video file."""
        # Create a temporary invalid video file
        with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_file:
            temp_path = Path(temp_file.name)

            # Create an analysis manager
            manager = AnalysisManager()
            manager.register_analyzer("hook")

            # Run the analysis and expect it to fail
            with pytest.raises(VideoProcessingError):
                await manager.analyze_video(temp_path)

    def test_cli_end_to_end(self, runner, test_video_path):
        """Test the CLI end-to-end workflow."""
        if not test_video_path.exists():
            pytest.skip(f"Test video not found at {test_video_path}")

        # Create a temporary output file
        with tempfile.NamedTemporaryFile(suffix=".json") as temp_file:
            output_path = Path(temp_file.name)

            # Run the CLI command
            result = runner.invoke(
                app, ["analyze", str(test_video_path), "--output", str(output_path)]
            )

            # Check that the command succeeded
            assert result.exit_code == 0, f"CLI command failed: {result.stdout}"
            assert "Analysis complete" in result.stdout, (
                "Output should indicate analysis is complete"
            )

            # Check that the output file was created
            assert output_path.exists(), "Output file should exist"

            # Check that the output file contains valid JSON
            with open(output_path) as f:
                report_data = json.load(f)
                assert "summary" in report_data, "Report should have a summary"
                assert "sections" in report_data, "Report should have sections"


class TestPerformanceBenchmarks:
    """Performance benchmarks for the video analyzer."""

    @pytest.mark.asyncio
    async def test_analyzer_performance(self, test_video_path):
        """Benchmark the performance of individual analyzers."""
        if not test_video_path.exists():
            pytest.skip(f"Test video not found at {test_video_path}")

        # Create an analysis manager
        manager = AnalysisManager()

        # Get all available analyzer types
        analyzer_types = AnalyzerRegistry.get_available_types()

        # Benchmark each analyzer individually
        performance_results = {}

        for analyzer_type in analyzer_types:
            # Create a fresh manager for each analyzer
            single_analyzer_manager = AnalysisManager()
            single_analyzer_manager.register_analyzer(analyzer_type)

            # Run the analysis and measure time
            start_time = time.time()
            results = await single_analyzer_manager.analyze_video(test_video_path)
            end_time = time.time()

            # Record the performance
            execution_time = end_time - start_time
            performance_results[analyzer_type] = execution_time

            # Verify the result
            assert analyzer_type in next(iter(results.keys())), (
                f"Result should be from {analyzer_type} analyzer"
            )

        # Print performance results
        print("\nAnalyzer Performance Benchmarks:")
        for analyzer_type, execution_time in sorted(
            performance_results.items(), key=lambda x: x[1]
        ):
            print(f"{analyzer_type}: {execution_time:.2f} seconds")

        # Calculate statistics
        total_time = sum(performance_results.values())
        avg_time = total_time / len(performance_results)
        max_time = max(performance_results.values())
        min_time = min(performance_results.values())

        print(f"\nTotal time: {total_time:.2f} seconds")
        print(f"Average time: {avg_time:.2f} seconds")
        print(
            f"Max time: {max_time:.2f} seconds (analyzer: {max(performance_results, key=performance_results.get)})"
        )
        print(
            f"Min time: {min_time:.2f} seconds (analyzer: {min(performance_results, key=performance_results.get)})"
        )

    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_performance(self, test_video_path):
        """Compare performance of parallel vs sequential analysis."""
        if not test_video_path.exists():
            pytest.skip(f"Test video not found at {test_video_path}")

        from video_analyzer.config.analysis_pipeline import AnalysisPipelineConfig

        # Create configurations for parallel and sequential analysis
        parallel_config = AnalysisPipelineConfig(parallel_analyzers=True)
        sequential_config = AnalysisPipelineConfig(parallel_analyzers=False)

        # Create managers with different configurations
        parallel_manager = AnalysisManager(pipeline_config=parallel_config)
        sequential_manager = AnalysisManager(pipeline_config=sequential_config)

        # Register the same analyzers for both managers
        analyzer_types = AnalyzerRegistry.get_available_types()
        for analyzer_type in analyzer_types:
            parallel_manager.register_analyzer(analyzer_type)
            sequential_manager.register_analyzer(analyzer_type)

        # Run parallel analysis
        start_time = time.time()
        parallel_results = await parallel_manager.analyze_video(test_video_path)
        parallel_time = time.time() - start_time

        # Run sequential analysis
        start_time = time.time()
        sequential_results = await sequential_manager.analyze_video(test_video_path)
        sequential_time = time.time() - start_time

        # Compare results
        assert len(parallel_results) == len(sequential_results), (
            "Both analyses should have the same number of results"
        )

        # Print performance comparison
        print("\nParallel vs Sequential Performance:")
        print(f"Parallel analysis: {parallel_time:.2f} seconds")
        print(f"Sequential analysis: {sequential_time:.2f} seconds")
        print(f"Speedup: {sequential_time / parallel_time:.2f}x")

        # Verify that parallel is faster (with some tolerance for small videos)
        if len(analyzer_types) > 1:  # Only expect speedup with multiple analyzers
            assert parallel_time <= sequential_time * 1.1, (
                "Parallel analysis should be faster than sequential"
            )


class TestVideoScenarios:
    """Tests for different video scenarios."""

    @pytest.fixture(autouse=True)
    def setup_test_videos(self):
        """Set up test videos for the tests."""
        # Import the create_test_videos function from the script
        from tests.create_test_videos import create_test_videos

        # Create the test videos
        create_test_videos()

    @pytest.mark.asyncio
    async def test_different_video_resolutions(self):
        """Test analysis with videos of different resolutions."""
        # Get paths to test videos with different resolutions
        low_res_path = Path("tests/test_data/low_res_video.mp4")
        high_res_path = Path("tests/test_data/high_res_video.mp4")

        # Skip if videos don't exist
        if not low_res_path.exists() or not high_res_path.exists():
            pytest.skip("Test videos not found")

        # Create an analysis manager
        manager = AnalysisManager()

        # Register analyzers that are sensitive to resolution
        manager.register_analyzer("visual")
        manager.register_analyzer("object")

        # Analyze low resolution video
        low_res_results = await manager.analyze_video(low_res_path)

        # Analyze high resolution video
        high_res_results = await manager.analyze_video(high_res_path)

        # Verify that both analyses completed successfully
        assert low_res_results, "Low resolution analysis should return results"
        assert high_res_results, "High resolution analysis should return results"

        # Compare the results (in a real test, we would check specific aspects)
        # For now, we just check that the analysis completed for both videos
        print("\nResolution Comparison:")
        print(f"Low resolution ({low_res_path}): {len(low_res_results)} results")
        print(f"High resolution ({high_res_path}): {len(high_res_results)} results")

    @pytest.mark.asyncio
    async def test_different_video_content(self):
        """Test analysis with different video content types."""
        # Get paths to test videos with different content
        basic_path = Path("tests/test_data/test_video.mp4")
        color_shift_path = Path("tests/test_data/color_shift_video.mp4")
        scene_change_path = Path("tests/test_data/scene_change_video.mp4")

        # Skip if videos don't exist
        if (
            not basic_path.exists()
            or not color_shift_path.exists()
            or not scene_change_path.exists()
        ):
            pytest.skip("Test videos not found")

        # Create an analysis manager
        manager = AnalysisManager()

        # Register analyzers that are sensitive to content type
        manager.register_analyzer("visual")
        manager.register_analyzer("progression")
        manager.register_analyzer("emotion")

        # Analyze videos with different content
        basic_results = await manager.analyze_video(basic_path)
        color_shift_results = await manager.analyze_video(color_shift_path)
        scene_change_results = await manager.analyze_video(scene_change_path)

        # Verify that all analyses completed successfully
        assert basic_results, "Basic video analysis should return results"
        assert color_shift_results, "Color shift video analysis should return results"
        assert scene_change_results, "Scene change video analysis should return results"

        # Compare the results (in a real test, we would check specific aspects)
        # For now, we just check that the analysis completed for all videos
        print("\nContent Type Comparison:")
        print(f"Basic video ({basic_path}): {len(basic_results)} results")
        print(
            f"Color shift video ({color_shift_path}): {len(color_shift_results)} results"
        )
        print(
            f"Scene change video ({scene_change_path}): {len(scene_change_results)} results"
        )

    @pytest.mark.asyncio
    async def test_different_video_durations(self):
        """Test analysis with videos of different durations."""
        # Get paths to test videos with different durations
        short_path = Path("tests/test_data/short_video.mp4")
        medium_path = Path("tests/test_data/medium_video.mp4")
        long_path = Path("tests/test_data/long_video.mp4")

        # Skip if videos don't exist
        if (
            not short_path.exists()
            or not medium_path.exists()
            or not long_path.exists()
        ):
            pytest.skip("Test videos not found")

        # Create an analysis manager
        manager = AnalysisManager()

        # Register analyzers
        manager.register_analyzer("hook")
        manager.register_analyzer("progression")

        # Analyze videos with different durations
        short_results = await manager.analyze_video(short_path)
        medium_results = await manager.analyze_video(medium_path)
        long_results = await manager.analyze_video(long_path)

        # Verify that all analyses completed successfully
        assert short_results, "Short video analysis should return results"
        assert medium_results, "Medium video analysis should return results"
        assert long_results, "Long video analysis should return results"

        # Compare execution times
        short_time = manager.execution_time

        # Reset manager and analyze medium video
        manager = AnalysisManager()
        manager.register_analyzer("hook")
        manager.register_analyzer("progression")
        await manager.analyze_video(medium_path)
        medium_time = manager.execution_time

        # Reset manager and analyze long video
        manager = AnalysisManager()
        manager.register_analyzer("hook")
        manager.register_analyzer("progression")
        await manager.analyze_video(long_path)
        long_time = manager.execution_time

        # Print execution times
        print("\nDuration Comparison:")
        print(f"Short video ({short_path}): {short_time:.2f} seconds")
        print(f"Medium video ({medium_path}): {medium_time:.2f} seconds")
        print(f"Long video ({long_path}): {long_time:.2f} seconds")

        # Verify that longer videos take more time to analyze
        # Note: This might not always be true due to various factors,
        # so we use a tolerance factor
        assert medium_time >= short_time * 0.8, (
            "Medium video should take longer than short video"
        )
        assert long_time >= medium_time * 0.8, (
            "Long video should take longer than medium video"
        )


# Helper function to create a test video using OpenCV
def create_test_video(
    output_path: Path,
    duration_seconds: int = 5,
    fps: int = 30,
    resolution: tuple = (640, 480),
):
    """
    Create a test video file for testing.

    Args:
        output_path: Path where the video will be saved
        duration_seconds: Duration of the video in seconds
        fps: Frames per second
        resolution: Video resolution (width, height)
    """
    import cv2
    import numpy as np

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, resolution)

    # Create frames
    total_frames = duration_seconds * fps
    for i in range(total_frames):
        # Create a frame with a moving rectangle
        frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)

        # Calculate position of the rectangle
        x = int((i / total_frames) * (resolution[0] - 100))
        y = int(resolution[1] / 2 - 50)

        # Draw a rectangle
        cv2.rectangle(frame, (x, y), (x + 100, y + 100), (0, 255, 0), -1)

        # Add frame number text
        cv2.putText(
            frame,
            f"Frame {i}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Write the frame
        out.write(frame)

    # Release the VideoWriter
    out.release()

    print(f"Created test video at {output_path}")


if __name__ == "__main__":
    # Create test videos if run directly
    test_video_dir = Path("tests/test_data")
    test_video_dir.mkdir(exist_ok=True)

    # Create a basic test video
    create_test_video(test_video_dir / "test_video.mp4")

    # Create videos with different durations
    create_test_video(test_video_dir / "short_video.mp4", duration_seconds=2)
    create_test_video(test_video_dir / "medium_video.mp4", duration_seconds=10)
    create_test_video(test_video_dir / "long_video.mp4", duration_seconds=30)

    print("Test videos created successfully")


class TestErrorHandlingAndCancellation:
    """Tests for error handling and cancellation."""

    @pytest.mark.asyncio
    async def test_analysis_cancellation(self, test_video_path):
        """Test cancellation of analysis."""
        if not test_video_path.exists():
            pytest.skip(f"Test video not found at {test_video_path}")

        # Create an analysis manager
        manager = AnalysisManager()

        # Register analyzers
        for analyzer_type in AnalyzerRegistry.get_available_types():
            manager.register_analyzer(analyzer_type)

        # Set up progress tracking
        progress_updates = []

        def on_progress(
            analyzer_id: str, progress: float, metadata: Dict[str, Any]
        ) -> None:
            progress_updates.append((analyzer_id, progress, metadata))

            # Cancel the analysis when we reach 30% progress
            if progress >= 0.3 and analyzer_id == "pipeline":
                cancellation_token.cancel()

        manager.set_progress_callback(on_progress)

        # Create a cancellation token
        cancellation_token = manager.create_cancellation_token()

        # Run the analysis and expect it to be cancelled
        from video_analyzer.analyzers.base import CancellationError

        with pytest.raises(CancellationError):
            await manager.analyze_video(test_video_path)

        # Verify that we received progress updates before cancellation
        assert len(progress_updates) > 0, (
            "There should be progress updates before cancellation"
        )

        # Verify that the last progress update indicates cancellation
        last_update = progress_updates[-1]
        assert last_update[2].get("status") == "cancelled", (
            "Last update should indicate cancellation"
        )

    @pytest.mark.asyncio
    async def test_error_recovery(self, test_video_path):
        """Test recovery from analyzer errors."""
        if not test_video_path.exists():
            pytest.skip(f"Test video not found at {test_video_path}")

        from video_analyzer.config.analysis_pipeline import AnalysisPipelineConfig

        # Create a pipeline configuration that doesn't fail on analyzer errors
        config = AnalysisPipelineConfig(fail_fast=False)

        # Create an analysis manager with the configuration
        manager = AnalysisManager(pipeline_config=config)

        # Register a mix of working analyzers and a mock failing analyzer
        manager.register_analyzer("hook")
        manager.register_analyzer("progression")

        # Create and register a failing analyzer
        from video_analyzer.analyzers.base import BaseAnalyzer, AnalysisResult

        class FailingAnalyzer(BaseAnalyzer):
            """Analyzer that always fails."""

            async def analyze(self, video_data):
                """Analyze method that raises an exception."""
                raise Exception("This analyzer always fails")

            @property
            def analyzer_id(self):
                """Get the analyzer ID."""
                return "failing_analyzer"

        manager._registered_analyzers["failing_analyzer"] = FailingAnalyzer()
        manager.pipeline.register_analyzer(FailingAnalyzer())

        # Run the analysis
        results = await manager.analyze_video(test_video_path)

        # Verify that we got results from the working analyzers
        assert "hook" in next(iter(results.keys())), (
            "Results should include hook analyzer"
        )
        assert "progression" in next(iter(results.keys())), (
            "Results should include progression analyzer"
        )

        # Verify that the failing analyzer is not in the results
        assert not any("failing" in analyzer_id for analyzer_id in results.keys()), (
            "Results should not include the failing analyzer"
        )

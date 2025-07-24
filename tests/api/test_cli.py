"""
Unit tests for the CLI interface.
"""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from video_analyzer.api.cli import app
from video_analyzer.analyzers.base import AnalyzerRegistry, CancellationError


@pytest.fixture
def runner():
    """Fixture for creating a CLI runner."""
    return CliRunner()


@pytest.fixture
def mock_video_path(tmp_path):
    """Create a mock video file for testing."""
    video_path = tmp_path / "test_video.mp4"
    # Create an empty file
    video_path.write_text("")
    return video_path


@pytest.fixture
def mock_analyzer_registry():
    """Mock the analyzer registry."""
    with patch("video_analyzer.api.cli.AnalyzerRegistry") as mock_registry:
        # Mock available analyzers
        mock_registry.get_available_types.return_value = [
            "hook",
            "progression",
            "visual",
            "audio",
        ]

        # Mock analyzer classes
        mock_analyzer = MagicMock()
        mock_analyzer.description = "Mock analyzer description"
        mock_analyzer.capabilities = ["Capability 1", "Capability 2"]

        mock_registry.get_analyzer_class.return_value = lambda: mock_analyzer

        yield mock_registry


@pytest.fixture
def mock_analysis_manager():
    """Mock the analysis manager."""
    with patch("video_analyzer.api.cli.AnalysisManager") as mock_manager_class:
        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager

        # Mock analyze_video method
        mock_results = {
            "hook": MagicMock(confidence=0.85, data={"key1": "value1"}),
            "progression": MagicMock(confidence=0.92, data={"key2": "value2"}),
        }
        mock_manager.analyze_video.return_value = mock_results

        # Mock generate_report method
        mock_report = MagicMock()
        mock_report.summary = "Test summary"
        mock_report.recommendations = ["Recommendation 1", "Recommendation 2"]
        mock_report.sections = {
            "Hook Analysis": {"confidence": 0.85, "key1": "value1"},
            "Progression Analysis": {"confidence": 0.92, "key2": "value2"},
        }
        mock_report.dict.return_value = {
            "summary": mock_report.summary,
            "recommendations": mock_report.recommendations,
            "sections": mock_report.sections,
        }
        mock_manager.generate_report.return_value = mock_report

        # Mock create_cancellation_token method
        mock_token = MagicMock()
        mock_manager.create_cancellation_token.return_value = mock_token

        yield mock_manager


def test_version_command(runner):
    """Test the version command."""
    with patch("importlib.metadata.version", return_value="1.0.0"):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "Video Analyzer version: 1.0.0" in result.stdout


def test_examples_command(runner):
    """Test the examples command."""
    result = runner.invoke(app, ["examples"])
    assert result.exit_code == 0
    assert "Video Analyzer CLI Examples" in result.stdout
    assert "Basic Usage" in result.stdout


def test_list_analyzers_command(runner, mock_analyzer_registry):
    """Test the list-analyzers command."""
    result = runner.invoke(app, ["list-analyzers"])
    assert result.exit_code == 0
    assert "Available Analyzers" in result.stdout
    assert "hook" in result.stdout
    assert "progression" in result.stdout
    assert "visual" in result.stdout
    assert "audio" in result.stdout


def test_list_analyzers_detailed_command(runner, mock_analyzer_registry):
    """Test the list-analyzers command with detailed flag."""
    result = runner.invoke(app, ["list-analyzers", "--detailed"])
    assert result.exit_code == 0
    assert "Available Analyzers" in result.stdout
    assert "Capability 1" in result.stdout
    assert "Capability 2" in result.stdout


def test_analyze_command_with_nonexistent_file(runner):
    """Test the analyze command with a nonexistent file."""
    # Since we're using exists=True in the Path argument, Typer will handle the error
    # and exit with a non-zero code, but the error message might not be in stdout
    result = runner.invoke(app, ["analyze", "nonexistent_file.mp4"])
    assert result.exit_code != 0


def test_analyze_command_basic(runner, mock_video_path, mock_analysis_manager):
    """Test the basic analyze command."""
    with patch("asyncio.get_event_loop"):
        result = runner.invoke(app, ["analyze", str(mock_video_path)])
        assert result.exit_code == 0
        assert "Analyzing video" in result.stdout
        assert "Analysis complete" in result.stdout
        assert "Test summary" in result.stdout
        assert "Recommendation 1" in result.stdout
        assert "Recommendation 2" in result.stdout


def test_analyze_command_with_output(
    runner, mock_video_path, mock_analysis_manager, tmp_path
):
    """Test the analyze command with output file."""
    output_path = tmp_path / "output.json"

    with patch("asyncio.get_event_loop"):
        result = runner.invoke(
            app, ["analyze", str(mock_video_path), "--output", str(output_path)]
        )

        assert result.exit_code == 0
        assert "Report saved to" in result.stdout
        assert output_path.exists()

        # Verify the content of the output file
        with open(output_path) as f:
            report_data = json.load(f)
            assert "summary" in report_data
            assert "recommendations" in report_data
            assert "sections" in report_data


def test_analyze_command_with_specific_analyzers(
    runner, mock_video_path, mock_analysis_manager
):
    """Test the analyze command with specific analyzers."""
    # Configure the mock to return both analyzers in the list
    mock_analysis_manager.registered_analyzers = ["hook", "progression"]

    with patch("asyncio.get_event_loop"):
        result = runner.invoke(
            app, ["analyze", str(mock_video_path), "--analyzers", "hook,progression"]
        )

        assert result.exit_code == 0
        assert "Analyzing video" in result.stdout

        # Verify that the manager was called with at least the hook analyzer
        mock_analysis_manager.register_analyzer.assert_any_call("hook")


def test_analyze_command_with_cancellation(
    runner, mock_video_path, mock_analysis_manager
):
    """Test the analyze command with cancellation."""
    # Make analyze_video raise a CancellationError
    mock_analysis_manager.analyze_video.side_effect = CancellationError("Cancelled")

    with patch("asyncio.get_event_loop"):
        result = runner.invoke(app, ["analyze", str(mock_video_path)])

        assert result.exit_code == 0
        assert "cancelled" in result.stdout.lower()


def test_analyze_command_with_error(runner, mock_video_path, mock_analysis_manager):
    """Test the analyze command with an error."""
    # Make analyze_video raise an exception
    mock_analysis_manager.analyze_video.side_effect = Exception("Test error")

    with patch("asyncio.get_event_loop"):
        result = runner.invoke(app, ["analyze", str(mock_video_path)])

        assert result.exit_code == 1
        assert "failed" in result.stdout.lower()
        assert "Test error" in result.stdout


def test_analyze_command_verbose(runner, mock_video_path, mock_analysis_manager):
    """Test the analyze command with verbose flag."""
    with patch("asyncio.get_event_loop"):
        result = runner.invoke(app, ["analyze", str(mock_video_path), "--verbose"])

        assert result.exit_code == 0
        assert "Analysis Configuration" in result.stdout


def test_html_report_generation(
    runner, mock_video_path, mock_analysis_manager, tmp_path
):
    """Test HTML report generation."""
    output_path = tmp_path / "output.html"

    with patch("asyncio.get_event_loop"):
        result = runner.invoke(
            app,
            [
                "analyze",
                str(mock_video_path),
                "--output",
                str(output_path),
                "--format",
                "html",
            ],
        )

        assert result.exit_code == 0
        assert "HTML report saved to" in result.stdout
        assert output_path.exists()

        # Verify the content of the HTML file
        with open(output_path) as f:
            html_content = f.read()
            assert "<!DOCTYPE html>" in html_content
            assert "Video Analysis Report" in html_content
            assert "Test summary" in html_content
            assert "Recommendation 1" in html_content
            assert "Recommendation 2" in html_content

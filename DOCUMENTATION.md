# Video Analyzer Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Analyzers](#analyzers)
6. [Data Models](#data-models)
7. [Configuration](#configuration)
8. [Error Handling](#error-handling)
9. [Performance Optimization](#performance-optimization)
10. [Development Guide](#development-guide)
11. [API Reference](#api-reference)
12. [Troubleshooting](#troubleshooting)

## Introduction

Video Analyzer is an AI-powered tool designed to analyze videos in great detail, providing comprehensive insights about various aspects of video content. Using computer vision and language model capabilities, the system breaks down videos into their constituent elements, analyzing hooks, progression, scenes, lighting, mood, storyline, and objects.

### Purpose

The primary goal of Video Analyzer is to help content creators learn from existing videos to create more engaging content in the future. By providing detailed analysis and actionable recommendations, it enables creators to understand what makes videos effective and how to improve their own content.

### Key Features

- Comprehensive video analysis across multiple dimensions
- AI-powered insights using computer vision and language models
- Detailed reports with actionable recommendations
- Support for various video formats
- Modular architecture for extensibility

## System Architecture

The Video Analyzer follows a modular architecture with the following high-level components:

```
CLI Interface → Video Processor → Frame Extractor → Analysis Pipeline → Report Generator
                                                     ↓
                                                  Analyzers
                                                     ↓
                                    Hook, Progression, Visual, Audio, Object, Emotion, Storytelling
```

### Core Components

1. **CLI Interface**: A Typer-based command-line interface for user interaction.
2. **Video Processor**: Handles video input, validation, and preprocessing.
3. **Frame Extractor**: Extracts frames from the video at appropriate intervals.
4. **Analysis Pipeline**: Orchestrates the various analyzers.
5. **Analyzers**: Specialized modules for different aspects of video analysis.
6. **Report Generator**: Compiles analysis results into a comprehensive report.

## Installation

### System Requirements

- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 1GB free space for installation, plus space for video files
- **GPU**: Optional but recommended for faster processing

### Installation Methods

#### Using pip

```bash
pip install video-analyzer
```

#### From Source

```bash
git clone https://github.com/yourusername/video-analyzer.git
cd video-analyzer
pip install -e .
```

### Verifying Installation

After installation, verify that the tool is working correctly:

```bash
video-analyzer --version
```

## Usage

### Basic Usage

```bash
video-analyzer analyze my_video.mp4
```

### Advanced Usage

```bash
video-analyzer analyze my_video.mp4 --analyzers hook,progression,visual --format html --output report.html --parallel --timeout 600 --verbose
```

### Command Reference

#### analyze

Analyze a video file and generate a detailed report.

```bash
video-analyzer analyze VIDEO_PATH [OPTIONS]
```

Options:

- `--format`, `-f`: Output format (json, html, pdf)
- `--output`, `-o`: Path to save the output report
- `--parallel/--sequential`: Run analyzers in parallel or sequentially
- `--analyzers`, `-a`: Specific analyzers to run (comma-separated list)
- `--timeout`, `-t`: Timeout for each analyzer in seconds
- `--verbose`, `-v`: Enable verbose output with detailed information

#### list-analyzers

List all available analyzers with descriptions.

```bash
video-analyzer list-analyzers [OPTIONS]
```

Options:

- `--detailed`, `-d`: Show detailed information about each analyzer

## Analyzers

The Video Analyzer includes several specialized analyzers, each focusing on a different aspect of video content:

### Hook Analyzer

Analyzes the hook section of the video (typically the first 5-15 seconds).

- Identifies hook techniques used to capture attention
- Evaluates hook effectiveness based on pacing, visuals, and audio
- Provides timestamps for key moments
- Compares hook strategies across different content

### Progression Analyzer

Analyzes the progression and structure of the video.

- Breaks down the video into distinct sections/segments
- Identifies pacing changes throughout the video
- Detects transitions between topics or scenes
- Evaluates narrative flow and coherence
- Provides insights on retention strategies

### Visual Elements Analyzer

Analyzes visual aspects of the video.

- Evaluates lighting techniques and quality
- Identifies color schemes and their emotional impact
- Detects camera movements and framing techniques
- Recognizes visual effects and their purpose
- Provides recommendations for visual improvements

### Audio Analyzer

Analyzes audio and speech elements of the video.

- Evaluates sound quality and clarity
- Assesses speech pacing, tone, and delivery style
- Identifies background music and its emotional effect
- Detects sound effects and their purpose
- Transcribes speech for further analysis

### Object Detector

Detects and analyzes objects, people, and brands in the video.

- Detects and identifies common objects
- Recognizes human faces and expressions
- Identifies brand logos and products
- Tracks screen time and positioning of objects
- Analyzes brand integration into content

### Emotion Analyzer

Analyzes the emotional impact and mood of the video.

- Identifies overall mood and tone
- Detects emotional shifts throughout the video
- Evaluates how visual and audio elements contribute to emotions
- Identifies techniques used to elicit specific emotions
- Creates an emotional journey map of the video

### Storytelling Analyzer

Analyzes narrative structure and storytelling techniques.

- Identifies narrative structure
- Detects character development or presenter techniques
- Evaluates conflict and resolution patterns
- Identifies persuasion techniques
- Provides insights on audience engagement strategies

## Data Models

The Video Analyzer uses several data models to represent video data and analysis results:

### VideoData

Represents the video being analyzed.

```python
class VideoData(BaseModel):
    path: Path
    frames: List[Frame]
    duration: float
    fps: float
    resolution: Tuple[int, int]
    metadata: Dict[str, Any]
```

### Frame

Represents a single frame from the video.

```python
class Frame(BaseModel):
    image: np.ndarray
    timestamp: float
    index: int
```

### AnalysisResult

Base class for all analysis results.

```python
class AnalysisResult(BaseModel):
    analyzer_id: str
    timestamp: datetime
    confidence: float
    data: Dict[str, Any]
```

### Report

Represents the final analysis report.

```python
class Report(BaseModel):
    video_id: str
    analysis_timestamp: datetime
    analysis_duration: float
    summary: str
    sections: Dict[str, Any]
    recommendations: List[str]
```

## Configuration

The Video Analyzer can be configured through several configuration classes:

### VideoProcessorConfig

Configures the video processor component.

```python
class VideoProcessorConfig(BaseModel):
    max_file_size_mb: int = 1000
    supported_formats: List[str] = ["mp4", "avi", "mov", "wmv"]
    validation_timeout_seconds: int = 30
```

### FrameExtractorConfig

Configures the frame extraction process.

```python
class FrameExtractorConfig(BaseModel):
    default_strategy: str = "uniform"
    uniform_frame_interval_seconds: float = 1.0
    scene_change_threshold: float = 0.3
    max_frames: int = 1000
```

### AnalysisPipelineConfig

Configures the analysis pipeline.

```python
class AnalysisPipelineConfig(BaseModel):
    parallel_analyzers: bool = True
    timeout_seconds: int = 300
    enabled_analyzers: Optional[List[str]] = None
```

## Error Handling

The Video Analyzer implements a comprehensive error handling strategy:

### Error Types

- **VideoAnalyzerError**: Base exception class for all errors
- **VideoProcessingError**: Errors related to video processing
- **AnalysisError**: Errors that occur during analysis
- **ReportGenerationError**: Errors that occur during report generation
- **ExternalServiceError**: Errors related to external services
- **CancellationError**: Raised when analysis is cancelled

### Error Recovery

The system attempts to recover from errors when possible:

- Individual analyzer failures don't stop the entire pipeline
- Fallback mechanisms for report generation
- Graceful degradation when external services fail

## Performance Optimization

The Video Analyzer includes several optimizations for performance:

### Asynchronous Processing

Uses async/await for I/O-bound operations, allowing multiple analyzers to run concurrently.

### Batch Processing

Processes frames in batches for more efficient analysis.

### Caching

Caches intermediate results to avoid redundant processing.

### Resource Management

Implements proper resource cleanup for video processing to minimize memory usage.

## Development Guide

### Adding a New Analyzer

To create a new analyzer:

1. Create a new class that inherits from `BaseAnalyzer`
2. Implement the required methods:
   - `analyze(self, video_data: VideoData) -> AnalysisResult`
   - `analyzer_id(self) -> str`
3. Register the analyzer using the `@AnalyzerRegistry.register` decorator

Example:

```python
@AnalyzerRegistry.register("my_analyzer")
class MyAnalyzer(BaseAnalyzer):
    async def analyze(self, video_data: VideoData) -> AnalysisResult:
        # Implementation
        return AnalysisResult(
            analyzer_id=self.analyzer_id,
            confidence=0.9,
            data={"key": "value"},
            video_id=str(video_data.path),
        )

    @property
    def analyzer_id(self) -> str:
        return "my_analyzer"
```

### Testing

Run the tests using pytest:

```bash
pytest
```

## API Reference

### video_analyzer.api

#### run()

Entry point for the CLI application.

### video_analyzer.analyzers

#### BaseAnalyzer

Abstract base class for all analyzers.

Methods:

- `analyze(self, video_data: VideoData) -> AnalysisResult`: Analyze the video data
- `analyze_with_progress(self, video_data: VideoData, progress_callback: Optional[AnalysisProgressCallback] = None, cancellation_token: Optional[AnalysisCancellationToken] = None) -> AnalysisResult`: Analyze with progress tracking and cancellation support

Properties:

- `analyzer_id`: Unique identifier for the analyzer
- `analyzer_type`: Type of the analyzer
- `analyzer_category`: Category of the analyzer
- `description`: Description of the analyzer
- `supports_cancellation`: Whether the analyzer supports cancellation
- `supports_progress`: Whether the analyzer supports progress reporting
- `required_frames`: Frame requirements for the analyzer
- `execution_time`: Execution time of the last analysis
- `metadata`: Metadata about the analyzer

#### AnalyzerRegistry

Registry for analyzer classes.

Methods:

- `register(analyzer_type: str)`: Decorator for registering analyzer classes
- `create(analyzer_type: str, config: Dict[str, Any] = None) -> BaseAnalyzer`: Create an analyzer of the specified type
- `get_available_types() -> List[str]`: Get a list of available analyzer types
- `get_analyzer_class(analyzer_type: str) -> Type[BaseAnalyzer]`: Get the analyzer class for the specified type

### video_analyzer.services

#### AnalysisManager

Manages the analysis process.

Methods:

- `register_analyzer(analyzer_type: str) -> None`: Register an analyzer
- `set_progress_callback(callback: Callable[[str, float, Dict[str, Any]], None]) -> None`: Set the progress callback
- `create_cancellation_token() -> AnalysisCancellationToken`: Create a cancellation token
- `analyze_video(video_path: Path) -> Dict[str, AnalysisResult]`: Analyze a video
- `generate_report(results: Dict[str, AnalysisResult], video_id: str) -> Report`: Generate a report from analysis results

#### AnalysisPipeline

Orchestrates the analysis pipeline.

Methods:

- `register_analyzer(analyzer: BaseAnalyzer) -> None`: Register an analyzer with the pipeline
- `run_analysis(video_data: VideoData) -> Dict[str, AnalysisResult]`: Run all registered analyzers on the video data

## Troubleshooting

### Common Issues

#### Video Format Not Supported

**Symptom**: Error message indicating that the video format is not supported.

**Solution**: Convert the video to a supported format (MP4, AVI, MOV, WMV) using a tool like FFmpeg:

```bash
ffmpeg -i input.unsupported output.mp4
```

#### Analysis Takes Too Long

**Symptom**: Analysis process takes a very long time to complete.

**Solution**:

- Use the `--timeout` option to set a shorter timeout for analyzers
- Specify only the analyzers you need using the `--analyzers` option
- Use a shorter video or trim the video to the relevant section

#### Out of Memory Error

**Symptom**: The process crashes with an out of memory error.

**Solution**:

- Process a smaller video
- Reduce the number of frames extracted by modifying the frame extractor configuration
- Run analyzers sequentially instead of in parallel using the `--sequential` flag

#### External Service Errors

**Symptom**: Errors related to external services like OpenAI API.

**Solution**:

- Check your API keys and ensure they are valid
- Check your internet connection
- Verify that you have sufficient quota/credits for the external service

### Getting Help

If you encounter issues not covered in this documentation, please:

1. Check the GitHub issues to see if the problem has been reported
2. Create a new issue with detailed information about the problem
3. Include the error message, command used, and system information

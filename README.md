# Video Analyzer

An AI-powered tool for comprehensive video content analysis. This tool analyzes videos in great detail, providing insights about various aspects of video content including hooks, progression, visual elements, audio, objects, emotional impact, and storytelling techniques.

## Overview

Video Analyzer is designed to help content creators learn from existing videos to create more engaging content. It leverages AI technologies including computer vision and language models to break down videos into their constituent elements and provide actionable insights.

## Features

- **Hook Analysis**: Identifies and analyzes video introductions, evaluating their effectiveness
- **Progression Analysis**: Breaks down video structure, pacing, and transitions
- **Visual Elements Analysis**: Evaluates lighting, color schemes, camera movements, and effects
- **Audio Analysis**: Assesses sound quality, speech patterns, music, and effects
- **Object and Brand Detection**: Identifies objects, faces, expressions, and brands in videos
- **Emotion Analysis**: Analyzes the mood and emotional impact of video content
- **Storytelling Analysis**: Evaluates narrative structure and engagement strategies
- **Comprehensive Reporting**: Generates detailed reports with actionable recommendations

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Install from PyPI

```bash
pip install video-analyzer
```

### Install from Source

```bash
git clone https://github.com/yourusername/video-analyzer.git
cd video-analyzer
pip install -e .
```

## Usage

### Command Line Interface

The Video Analyzer provides a command-line interface for analyzing videos:

```bash
# Analyze a video with all available analyzers
video-analyzer analyze my_video.mp4

# Analyze a video with specific analyzers
video-analyzer analyze my_video.mp4 --analyzers hook,progression,visual

# Save analysis results to a file
video-analyzer analyze my_video.mp4 --output report.json

# Generate an HTML report
video-analyzer analyze my_video.mp4 --format html --output report.html

# List all available analyzers
video-analyzer list-analyzers

# Show detailed information about available analyzers
video-analyzer list-analyzers --detailed
```

### Options

- `--format`, `-f`: Output format (json, html, pdf)
- `--output`, `-o`: Path to save the output report
- `--parallel/--sequential`: Run analyzers in parallel or sequentially
- `--analyzers`, `-a`: Specific analyzers to run (comma-separated list)
- `--timeout`, `-t`: Timeout for each analyzer in seconds
- `--verbose`, `-v`: Enable verbose output with detailed information

## Supported Video Formats

The Video Analyzer supports the following video formats:

- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- WMV (.wmv)

## Technical Architecture

The system follows a modular architecture with the following high-level components:

1. **CLI Interface**: A Typer-based command-line interface for user interaction
2. **Video Processor**: Handles video input, validation, and preprocessing
3. **Frame Extractor**: Extracts frames from the video at appropriate intervals
4. **Analysis Pipeline**: Orchestrates the various analyzers
5. **Analyzers**: Specialized modules for different aspects of video analysis
6. **Report Generator**: Compiles analysis results into a comprehensive report

### Analyzer Types

- **Hook Analyzer**: Identifies and analyzes video introductions
- **Progression Analyzer**: Analyzes video structure and pacing
- **Visual Elements Analyzer**: Evaluates lighting, color schemes, and camera techniques
- **Audio Analyzer**: Assesses sound quality and speech patterns
- **Object Detector**: Identifies objects, faces, and brands in videos
- **Emotion Analyzer**: Analyzes the mood and emotional impact
- **Storytelling Analyzer**: Evaluates narrative structure and engagement strategies

## Data Flow

1. **Input**: User provides a video file through the CLI
2. **Validation**: The video processor validates the file format and size
3. **Frame Extraction**: The frame extractor pulls relevant frames from the video
4. **Analysis**: Multiple analyzers process the video data in parallel or sequentially
5. **Report Generation**: Results are compiled into a comprehensive report
6. **Output**: The report is presented to the user in the requested format

## Requirements

See the `requirements.txt` file for a complete list of dependencies.

## Error Handling

The Video Analyzer implements comprehensive error handling:

- **Input Validation Errors**: Clear error messages for invalid inputs
- **Processing Errors**: Graceful handling of video processing issues
- **Analysis Errors**: Individual analyzer failures don't stop the entire pipeline
- **External Service Errors**: Retry mechanisms and graceful degradation

## Performance Considerations

- **Asynchronous Processing**: Uses async/await for I/O-bound operations
- **Batch Processing**: Processes frames in batches for efficiency
- **Caching**: Caches intermediate results to avoid redundant processing
- **Resource Management**: Implements proper resource cleanup for video processing

## Security Considerations

- **Input Validation**: Validates all user inputs to prevent injection attacks
- **File Handling**: Implements secure file handling practices
- **Rate Limiting**: Implements rate limiting to prevent abuse

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

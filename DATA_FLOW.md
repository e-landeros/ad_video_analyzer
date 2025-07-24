# Video Analyzer Data Flow

## System Architecture Diagram

```mermaid
graph TD
    A[User] -->|Video File| B[CLI Interface]
    B -->|Video Path| C[Video Processor]
    C -->|Validated Video| D[Frame Extractor]
    D -->|Frames| E[Analysis Pipeline]

    E -->|Video Data| F1[Hook Analyzer]
    E -->|Video Data| F2[Progression Analyzer]
    E -->|Video Data| F3[Visual Elements Analyzer]
    E -->|Video Data| F4[Audio Analyzer]
    E -->|Video Data| F5[Object Detector]
    E -->|Video Data| F6[Emotion Analyzer]
    E -->|Video Data| F7[Storytelling Analyzer]

    F1 -->|Analysis Results| G[Report Generator]
    F2 -->|Analysis Results| G
    F3 -->|Analysis Results| G
    F4 -->|Analysis Results| G
    F5 -->|Analysis Results| G
    F6 -->|Analysis Results| G
    F7 -->|Analysis Results| G

    G -->|Report| H[Output]
    H -->|JSON/HTML/PDF| A

    I[OpenAI API] -.->|LLM Analysis| F1
    I -.->|LLM Analysis| F6
    I -.->|LLM Analysis| F7

    J[Computer Vision] -.->|Object Detection| F5
    J -.->|Visual Analysis| F3

    K[Audio Processing] -.->|Speech Analysis| F4
```

## Detailed Data Flow

1. **User Input**

   - User provides a video file path via CLI
   - Optional parameters: output format, analyzers to use, etc.

2. **Video Processing**

   - Video Processor validates the file format and size
   - Video metadata is extracted (duration, fps, resolution)

3. **Frame Extraction**

   - Frames are extracted using the specified strategy:
     - Uniform: Extract frames at regular intervals
     - Scene Change: Extract frames at scene changes
     - Keyframe: Extract keyframes

4. **Analysis Pipeline**

   - VideoData object is created with video metadata and frames
   - Analysis tasks are created for each enabled analyzer
   - Tasks are executed in parallel or sequentially based on configuration

5. **Individual Analyzers**

   - Each analyzer processes the video data independently
   - Some analyzers may depend on results from other analyzers
   - External services (OpenAI API, computer vision) may be used
   - Progress is reported back to the pipeline

6. **Report Generation**

   - Analysis results are collected from all analyzers
   - Results are organized into sections
   - Summary and recommendations are generated
   - Report is formatted according to the requested output format

7. **Output**
   - Report is either displayed to the user or saved to a file
   - Output formats: JSON, HTML, PDF

## Data Models Flow

```mermaid
classDiagram
    VideoData "1" --> "*" Frame
    AnalysisResult <|-- HookAnalysisResult
    AnalysisResult <|-- ProgressionAnalysisResult
    AnalysisResult <|-- VisualAnalysisResult
    AnalysisResult <|-- AudioAnalysisResult
    AnalysisResult <|-- ObjectDetectionResult
    AnalysisResult <|-- EmotionAnalysisResult
    AnalysisResult <|-- StorytellingAnalysisResult
    Report "1" --> "*" AnalysisResult

    class VideoData {
        +Path path
        +List~Frame~ frames
        +float duration
        +float fps
        +Tuple~int,int~ resolution
        +Dict metadata
    }

    class Frame {
        +ndarray image
        +float timestamp
        +int index
    }

    class AnalysisResult {
        +str analyzer_id
        +datetime timestamp
        +float confidence
        +Dict data
    }

    class Report {
        +str video_id
        +datetime analysis_timestamp
        +float analysis_duration
        +str summary
        +Dict sections
        +List~str~ recommendations
    }
```

## Process Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant VideoProcessor
    participant FrameExtractor
    participant AnalysisPipeline
    participant Analyzers
    participant ReportGenerator

    User->>CLI: video-analyzer analyze video.mp4
    CLI->>VideoProcessor: process_video(video.mp4)
    VideoProcessor->>VideoProcessor: validate_video()
    VideoProcessor->>FrameExtractor: extract_frames()
    FrameExtractor-->>VideoProcessor: frames
    VideoProcessor-->>CLI: video_data

    CLI->>AnalysisPipeline: run_analysis(video_data)

    par Parallel Analysis
        AnalysisPipeline->>Analyzers: analyze_hook()
        Analyzers-->>AnalysisPipeline: hook_result

        AnalysisPipeline->>Analyzers: analyze_progression()
        Analyzers-->>AnalysisPipeline: progression_result

        AnalysisPipeline->>Analyzers: analyze_visual()
        Analyzers-->>AnalysisPipeline: visual_result

        AnalysisPipeline->>Analyzers: analyze_audio()
        Analyzers-->>AnalysisPipeline: audio_result

        AnalysisPipeline->>Analyzers: analyze_objects()
        Analyzers-->>AnalysisPipeline: object_result

        AnalysisPipeline->>Analyzers: analyze_emotion()
        Analyzers-->>AnalysisPipeline: emotion_result

        AnalysisPipeline->>Analyzers: analyze_storytelling()
        Analyzers-->>AnalysisPipeline: storytelling_result
    end

    AnalysisPipeline-->>CLI: all_results

    CLI->>ReportGenerator: generate_report(all_results)
    ReportGenerator-->>CLI: report

    alt Output to File
        CLI->>User: Save report to file
    else Display in Console
        CLI->>User: Display report summary
    end
```

## Error Handling Flow

```mermaid
flowchart TD
    A[Start Analysis] --> B{Valid Video?}
    B -->|Yes| C[Extract Frames]
    B -->|No| D[Video Processing Error]

    C --> E{Frames Extracted?}
    E -->|Yes| F[Run Analyzers]
    E -->|No| G[Frame Extraction Error]

    F --> H{All Analyzers Complete?}
    H -->|Yes| I[Generate Report]
    H -->|No| J{Some Analyzers Failed?}

    J -->|Yes| K[Partial Results]
    J -->|No| L[Analysis Error]

    K --> I

    I --> M{Report Generated?}
    M -->|Yes| N[Return Report]
    M -->|No| O[Report Generation Error]

    D --> P[Error Handling]
    G --> P
    L --> P
    O --> P

    P --> Q{Recoverable?}
    Q -->|Yes| R[Recovery Strategy]
    Q -->|No| S[Fatal Error]

    R --> T[Return Partial Results]
    S --> U[Exit with Error]
```

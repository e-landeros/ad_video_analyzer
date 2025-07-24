"""
Report generator for creating comprehensive analysis reports.

This module provides functionality for generating detailed reports from analysis results
in various formats (JSON, HTML, PDF).
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import base64
import io
import numpy as np
from PIL import Image


# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


# Import HTML class for PDF generation (used in _save_pdf_report)
try:
    from weasyprint import HTML
except ImportError:
    # Define a placeholder class for tests that mock this
    class HTML:
        def __init__(self, string=None):
            self.string = string

        def write_pdf(self, output_path):
            raise ImportError(
                "PDF generation requires weasyprint. Install with: pip install weasyprint"
            )


from video_analyzer.models.video import VideoData, Frame
from video_analyzer.models.analysis import (
    AnalysisResult,
    Report,
    HookAnalysisResult,
    ProgressionAnalysisResult,
    VisualAnalysisResult,
    AudioAnalysisResult,
    ObjectDetectionResult,
    EmotionAnalysisResult,
    StorytellingAnalysisResult,
)

# Set up logging
logger = logging.getLogger(__name__)


class ReportGeneratorConfig:
    """
    Configuration for the report generator.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        include_visual_examples: bool = True,
        max_visual_examples: int = 5,
        visual_example_quality: int = 80,
        visual_example_max_size: Tuple[int, int] = (800, 600),
    ):
        """
        Initialize the report generator configuration.

        Args:
            output_dir: Directory to save reports to (default: current directory)
            include_visual_examples: Whether to include visual examples in the report
            max_visual_examples: Maximum number of visual examples to include
            visual_example_quality: JPEG quality for visual examples (0-100)
            visual_example_max_size: Maximum size for visual examples (width, height)
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.include_visual_examples = include_visual_examples
        self.max_visual_examples = max_visual_examples
        self.visual_example_quality = visual_example_quality
        self.visual_example_max_size = visual_example_max_size


class ReportGenerator:
    """
    Generates comprehensive reports from analysis results.

    This class takes analysis results from various analyzers and compiles them into
    a comprehensive report in different formats (JSON, HTML, PDF).
    """

    def __init__(self, config: Optional[ReportGeneratorConfig] = None):
        """
        Initialize the report generator.

        Args:
            config: Configuration for the report generator
        """
        self.config = config or ReportGeneratorConfig()
        logger.debug("Initialized ReportGenerator")

    def generate_report(
        self,
        results: Dict[str, AnalysisResult],
        video_data: VideoData,
        analysis_duration: float,
    ) -> Report:
        """
        Generate a comprehensive report from analysis results.

        Args:
            results: Analysis results from different analyzers
            video_data: Video data that was analyzed
            analysis_duration: Duration of the analysis in seconds

        Returns:
            Report: Comprehensive analysis report
        """
        logger.info(f"Generating report for video: {video_data.path}")

        # Create a unique video ID
        video_id = f"{video_data.path.stem}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Create sections for each analyzer type
        sections = self._create_report_sections(results)

        # Create a summary
        summary = self._create_report_summary(results, video_data)

        # Collect recommendations from all analyzers
        recommendations = self._collect_recommendations(results)

        # Extract visual examples if configured
        visual_examples = []
        if self.config.include_visual_examples:
            visual_examples = self._extract_visual_examples(results, video_data)

        # Create the report
        # Ensure we have all required sections
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
            if section not in sections and section + "s" in sections:
                # Handle plural form (e.g., "object" -> "objects")
                sections[section] = sections[section + "s"]
            elif section not in sections:
                # Add empty section if missing
                sections[section] = {}

        report = Report(
            video_id=video_id,
            analysis_timestamp=datetime.now(),
            analysis_duration=analysis_duration,
            summary=summary,
            sections=sections,
            recommendations=recommendations,
            visual_examples=visual_examples,
        )

        logger.info(f"Report generated for video: {video_data.path}")
        return report

    def save_report(
        self, report: Report, format: str = "json", output_path: Optional[Path] = None
    ) -> Path:
        """
        Save the report to a file in the specified format.

        Args:
            report: The report to save
            format: Output format (json, html, pdf)
            output_path: Path to save the report to (default: auto-generated)

        Returns:
            Path: Path to the saved report file

        Raises:
            ValueError: If the format is not supported
        """
        # Create output directory if it doesn't exist
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"report_{report.video_id}_{timestamp}.{format.lower()}"
            output_path = self.config.output_dir / filename

        # Save the report in the specified format
        if format.lower() == "json":
            return self._save_json_report(report, output_path)
        elif format.lower() == "html":
            return self._save_html_report(report, output_path)
        elif format.lower() == "pdf":
            return self._save_pdf_report(report, output_path)
        else:
            raise ValueError(f"Unsupported report format: {format}")

    def _save_json_report(self, report: Report, output_path: Path) -> Path:
        """
        Save the report as a JSON file.

        Args:
            report: The report to save
            output_path: Path to save the report to

        Returns:
            Path: Path to the saved report file
        """
        # Convert the report to a dictionary
        report_dict = (
            report.model_dump() if hasattr(report, "model_dump") else report.dict()
        )

        # Convert datetime objects to ISO format strings
        report_dict["analysis_timestamp"] = report_dict[
            "analysis_timestamp"
        ].isoformat()

        # Save the report as JSON
        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2, cls=DateTimeEncoder)

        logger.info(f"JSON report saved to: {output_path}")
        return output_path

    def _save_html_report(self, report: Report, output_path: Path) -> Path:
        """
        Save the report as an HTML file.

        Args:
            report: The report to save
            output_path: Path to save the report to

        Returns:
            Path: Path to the saved report file
        """
        # Generate HTML content
        html_content = self._generate_html_report(report)

        # Save the report as HTML
        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"HTML report saved to: {output_path}")
        return output_path

    def _save_pdf_report(self, report: Report, output_path: Path) -> Path:
        """
        Save the report as a PDF file.

        Args:
            report: The report to save
            output_path: Path to save the report to

        Returns:
            Path: Path to the saved report file

        Note:
            This method requires additional dependencies (weasyprint) to be installed.
        """
        # Generate HTML content
        html_content = self._generate_html_report(report)

        # Convert HTML to PDF
        HTML(string=html_content).write_pdf(output_path)

        logger.info(f"PDF report saved to: {output_path}")
        return output_path

        # Generate HTML content
        html_content = self._generate_html_report(report)

        # Convert HTML to PDF
        HTML(string=html_content).write_pdf(output_path)

        logger.info(f"PDF report saved to: {output_path}")
        return output_path

    def _generate_html_report(self, report: Report) -> str:
        """
        Generate HTML content for the report.

        Args:
            report: The report to generate HTML for

        Returns:
            str: HTML content
        """
        # Create a basic HTML template
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Video Analysis Report: {report.video_id}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .summary {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .recommendations {{
                    background-color: #e8f4f8;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .section {{
                    margin-bottom: 30px;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 20px;
                }}
                .visual-examples {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 15px;
                    margin-top: 20px;
                }}
                .visual-example {{
                    max-width: 300px;
                    margin-bottom: 15px;
                }}
                .visual-example img {{
                    max-width: 100%;
                    border-radius: 5px;
                }}
                .visual-example .caption {{
                    font-size: 0.9em;
                    color: #666;
                    margin-top: 5px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
            </style>
        </head>
        <body>
            <h1>Video Analysis Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p>{report.summary}</p>
                <p><strong>Analysis Date:</strong> {report.analysis_timestamp.strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p><strong>Analysis Duration:</strong> {report.analysis_duration:.2f} seconds</p>
            </div>
            
            <div class="recommendations">
                <h2>Recommendations</h2>
                <ul>
        """

        # Add recommendations
        for recommendation in report.recommendations:
            html += f"<li>{recommendation}</li>\n"

        html += """
                </ul>
            </div>
        """

        # Add sections
        for section_name, section_data in report.sections.items():
            html += f"""
            <div class="section">
                <h2>{section_name.capitalize()} Analysis</h2>
            """

            # Add section content based on section type
            if section_name == "hook":
                html += self._generate_hook_section_html(section_data)
            elif section_name == "progression":
                html += self._generate_progression_section_html(section_data)
            elif section_name == "visual":
                html += self._generate_visual_section_html(section_data)
            elif section_name == "audio":
                html += self._generate_audio_section_html(section_data)
            elif section_name == "object":
                html += self._generate_object_section_html(section_data)
            elif section_name == "emotion":
                html += self._generate_emotion_section_html(section_data)
            elif section_name == "storytelling":
                html += self._generate_storytelling_section_html(section_data)
            else:
                # Generic section content
                html += f"<pre>{json.dumps(section_data, indent=2)}</pre>"

            html += """
            </div>
            """

        # Add visual examples
        if report.visual_examples:
            html += """
            <div class="section">
                <h2>Visual Examples</h2>
                <div class="visual-examples">
            """

            for example in report.visual_examples:
                if "image_data" in example:
                    html += f"""
                    <div class="visual-example">
                        <img src="data:image/jpeg;base64,{example["image_data"]}" alt="{example["description"]}">
                        <div class="caption">
                            <strong>{example["category"]}:</strong> {example["description"]} (Time: {example["timestamp"]:.2f}s)
                        </div>
                    </div>
                    """

            html += """
                </div>
            </div>
            """

        # Close HTML
        html += """
        </body>
        </html>
        """

        return html

    def _generate_hook_section_html(self, section_data: Dict[str, Any]) -> str:
        """Generate HTML content for hook analysis section."""
        html = f"""
        <p><strong>Hook Duration:</strong> {section_data.get("hook_start_time", 0):.2f}s - {section_data.get("hook_end_time", 0):.2f}s</p>
        <p><strong>Hook Effectiveness:</strong> {section_data.get("hook_effectiveness", 0) * 100:.1f}%</p>
        
        <h3>Hook Techniques</h3>
        <ul>
        """

        for technique in section_data.get("hook_techniques", []):
            html += f"<li>{technique}</li>\n"

        html += """
        </ul>
        
        <h3>Key Moments</h3>
        <table>
            <tr>
                <th>Time</th>
                <th>Description</th>
            </tr>
        """

        for moment in section_data.get("key_moments", []):
            html += f"""
            <tr>
                <td>{moment.get("timestamp", 0):.2f}s</td>
                <td>{moment.get("description", "")}</td>
            </tr>
            """

        html += """
        </table>
        """

        return html

    def _generate_progression_section_html(self, section_data: Dict[str, Any]) -> str:
        """Generate HTML content for progression analysis section."""
        html = f"""
        <p><strong>Narrative Flow Score:</strong> {section_data.get("narrative_flow_score", 0) * 100:.1f}%</p>
        
        <h3>Video Sections</h3>
        <table>
            <tr>
                <th>Time Range</th>
                <th>Title</th>
                <th>Description</th>
            </tr>
        """

        for section in section_data.get("sections", []):
            html += f"""
            <tr>
                <td>{section.get("start_time", 0):.2f}s - {section.get("end_time", 0):.2f}s</td>
                <td>{section.get("title", "")}</td>
                <td>{section.get("description", "")}</td>
            </tr>
            """

        html += """
        </table>
        
        <h3>Transitions</h3>
        <table>
            <tr>
                <th>Time</th>
                <th>Type</th>
                <th>Description</th>
            </tr>
        """

        for transition in section_data.get("transitions", []):
            html += f"""
            <tr>
                <td>{transition.get("timestamp", 0):.2f}s</td>
                <td>{transition.get("type", "")}</td>
                <td>{transition.get("description", "")}</td>
            </tr>
            """

        html += """
        </table>
        
        <h3>Retention Strategies</h3>
        <ul>
        """

        for strategy in section_data.get("retention_strategies", []):
            html += f"<li>{strategy}</li>\n"

        html += """
        </ul>
        """

        return html

    def _generate_visual_section_html(self, section_data: Dict[str, Any]) -> str:
        """Generate HTML content for visual analysis section."""
        html = f"""
        <p><strong>Lighting Quality:</strong> {section_data.get("lighting_quality", 0) * 100:.1f}%</p>
        
        <h3>Color Schemes</h3>
        <table>
            <tr>
                <th>Time</th>
                <th>Colors</th>
                <th>Mood</th>
            </tr>
        """

        for scheme in section_data.get("color_schemes", []):
            colors = ", ".join(scheme.get("colors", []))
            html += f"""
            <tr>
                <td>{scheme.get("timestamp", 0):.2f}s</td>
                <td>{colors}</td>
                <td>{scheme.get("mood", "")}</td>
            </tr>
            """

        html += """
        </table>
        
        <h3>Camera Movements</h3>
        <table>
            <tr>
                <th>Time</th>
                <th>Type</th>
                <th>Duration</th>
            </tr>
        """

        for movement in section_data.get("camera_movements", []):
            html += f"""
            <tr>
                <td>{movement.get("timestamp", 0):.2f}s</td>
                <td>{movement.get("type", "")}</td>
                <td>{movement.get("duration", 0):.2f}s</td>
            </tr>
            """

        html += """
        </table>
        
        <h3>Visual Effects</h3>
        <table>
            <tr>
                <th>Time</th>
                <th>Type</th>
                <th>Purpose</th>
            </tr>
        """

        for effect in section_data.get("visual_effects", []):
            html += f"""
            <tr>
                <td>{effect.get("timestamp", 0):.2f}s</td>
                <td>{effect.get("type", "")}</td>
                <td>{effect.get("purpose", "")}</td>
            </tr>
            """

        html += """
        </table>
        
        <h3>Visual Recommendations</h3>
        <ul>
        """

        for recommendation in section_data.get("visual_recommendations", []):
            html += f"<li>{recommendation}</li>\n"

        html += """
        </ul>
        """

        return html

    def _generate_audio_section_html(self, section_data: Dict[str, Any]) -> str:
        """Generate HTML content for audio analysis section."""
        html = f"""
        <p><strong>Sound Quality:</strong> {section_data.get("sound_quality", 0) * 100:.1f}%</p>
        
        <h3>Speech Analysis</h3>
        """

        speech_analysis = section_data.get("speech_analysis", {})
        if speech_analysis:
            html += "<ul>"
            for key, value in speech_analysis.items():
                html += f"<li><strong>{key.capitalize()}:</strong> {value}</li>\n"
            html += "</ul>"

        html += """
        <h3>Background Music</h3>
        <table>
            <tr>
                <th>Time Range</th>
                <th>Mood</th>
                <th>Description</th>
            </tr>
        """

        for music in section_data.get("background_music", []):
            html += f"""
            <tr>
                <td>{music.get("start_time", 0):.2f}s - {music.get("end_time", 0):.2f}s</td>
                <td>{music.get("mood", "")}</td>
                <td>{music.get("description", "")}</td>
            </tr>
            """

        html += """
        </table>
        
        <h3>Sound Effects</h3>
        <table>
            <tr>
                <th>Time</th>
                <th>Type</th>
                <th>Purpose</th>
            </tr>
        """

        for effect in section_data.get("sound_effects", []):
            html += f"""
            <tr>
                <td>{effect.get("timestamp", 0):.2f}s</td>
                <td>{effect.get("type", "")}</td>
                <td>{effect.get("purpose", "")}</td>
            </tr>
            """

        html += """
        </table>
        
        <h3>Transcription</h3>
        <div style="max-height: 300px; overflow-y: auto; background-color: #f9f9f9; padding: 10px; border-radius: 5px;">
            <pre>{section_data.get('transcription', 'No transcription available.')}</pre>
        </div>
        """

        return html

    def _generate_object_section_html(self, section_data: Dict[str, Any]) -> str:
        """Generate HTML content for object detection section."""
        html = f"""
        <p><strong>Brand Integration Score:</strong> {section_data.get("brand_integration_score", 0) * 100:.1f}%</p>
        
        <h3>Screen Time Analysis</h3>
        """

        screen_time = section_data.get("screen_time_analysis", {})
        if screen_time:
            html += "<table><tr><th>Object</th><th>Screen Time</th></tr>"
            for obj, time in screen_time.get("by_label", {}).items():
                html += f"<tr><td>{obj}</td><td>{time:.2f}s</td></tr>"
            html += "</table>"

        html += """
        <h3>Detected Objects</h3>
        <table>
            <tr>
                <th>Time</th>
                <th>Label</th>
                <th>Confidence</th>
            </tr>
        """

        # Limit to top 20 objects to avoid overwhelming the report
        objects = section_data.get("objects", [])[:20]
        for obj in objects:
            html += f"""
            <tr>
                <td>{obj.get("timestamp", 0):.2f}s</td>
                <td>{obj.get("label", "")}</td>
                <td>{obj.get("confidence", 0) * 100:.1f}%</td>
            </tr>
            """

        html += """
        </table>
        
        <h3>Detected Faces</h3>
        <table>
            <tr>
                <th>Time</th>
                <th>Expression</th>
                <th>Confidence</th>
            </tr>
        """

        # Limit to top 20 faces to avoid overwhelming the report
        faces = section_data.get("faces", [])[:20]
        for face in faces:
            html += f"""
            <tr>
                <td>{face.get("timestamp", 0):.2f}s</td>
                <td>{face.get("expression", "")}</td>
                <td>{face.get("confidence", 0) * 100:.1f}%</td>
            </tr>
            """

        html += """
        </table>
        
        <h3>Detected Brands</h3>
        <table>
            <tr>
                <th>Time</th>
                <th>Brand</th>
                <th>Screen Position</th>
            </tr>
        """

        for brand in section_data.get("brands", []):
            position = f"({brand.get('bounding_box', [0, 0, 0, 0])[0]}, {brand.get('bounding_box', [0, 0, 0, 0])[1]})"
            html += f"""
            <tr>
                <td>{brand.get("timestamp", 0):.2f}s</td>
                <td>{brand.get("name", "")}</td>
                <td>{position}</td>
            </tr>
            """

        html += """
        </table>
        """

        return html

    def _generate_emotion_section_html(self, section_data: Dict[str, Any]) -> str:
        """Generate HTML content for emotion analysis section."""
        html = f"""
        <p><strong>Overall Mood:</strong> {section_data.get("overall_mood", "")}</p>
        
        <h3>Emotional Shifts</h3>
        <table>
            <tr>
                <th>Time</th>
                <th>From</th>
                <th>To</th>
                <th>Trigger</th>
            </tr>
        """

        for shift in section_data.get("emotional_shifts", []):
            html += f"""
            <tr>
                <td>{shift.get("timestamp", 0):.2f}s</td>
                <td>{shift.get("from_emotion", "")}</td>
                <td>{shift.get("to_emotion", "")}</td>
                <td>{shift.get("trigger", "")}</td>
            </tr>
            """

        html += """
        </table>
        
        <h3>Emotional Elements</h3>
        """

        elements = section_data.get("emotional_elements", {})
        if elements:
            html += "<h4>Visual Elements</h4><ul>"
            for element in elements.get("visual", []):
                html += f"<li>{element}</li>\n"
            html += "</ul>"

            html += "<h4>Audio Elements</h4><ul>"
            for element in elements.get("audio", []):
                html += f"<li>{element}</li>\n"
            html += "</ul>"

        html += """
        <h3>Emotion Techniques</h3>
        <ul>
        """

        for technique in section_data.get("emotion_techniques", []):
            html += f"<li>{technique}</li>\n"

        html += """
        </ul>
        
        <h3>Emotional Journey</h3>
        <table>
            <tr>
                <th>Time</th>
                <th>Emotion</th>
                <th>Intensity</th>
            </tr>
        """

        for entry in section_data.get("emotional_journey", []):
            html += f"""
            <tr>
                <td>{entry.get("timestamp", 0):.2f}s</td>
                <td>{entry.get("emotion", "")}</td>
                <td>{entry.get("intensity", 0) * 100:.1f}%</td>
            </tr>
            """

        html += """
        </table>
        """

        return html

    def _generate_storytelling_section_html(self, section_data: Dict[str, Any]) -> str:
        """Generate HTML content for storytelling analysis section."""
        html = f"""
        <p><strong>Narrative Structure:</strong> {section_data.get("narrative_structure", "")}</p>
        
        <h3>Character Development</h3>
        <table>
            <tr>
                <th>Time</th>
                <th>Character</th>
                <th>Development Type</th>
                <th>Description</th>
            </tr>
        """

        for entry in section_data.get("character_development", []):
            html += f"""
            <tr>
                <td>{entry.get("timestamp", 0):.2f}s</td>
                <td>{entry.get("character", "")}</td>
                <td>{entry.get("development_type", "")}</td>
                <td>{entry.get("description", "")}</td>
            </tr>
            """

        html += """
        </table>
        
        <h3>Conflict Patterns</h3>
        <table>
            <tr>
                <th>Conflict Time</th>
                <th>Resolution Time</th>
                <th>Conflict Type</th>
                <th>Description</th>
            </tr>
        """

        for pattern in section_data.get("conflict_patterns", []):
            html += f"""
            <tr>
                <td>{pattern.get("conflict_timestamp", 0):.2f}s</td>
                <td>{pattern.get("resolution_timestamp", 0):.2f}s</td>
                <td>{pattern.get("conflict_type", "")}</td>
                <td>{pattern.get("description", "")}</td>
            </tr>
            """

        html += """
        </table>
        
        <h3>Persuasion Techniques</h3>
        <ul>
        """

        for technique in section_data.get("persuasion_techniques", []):
            html += f"<li>{technique}</li>\n"

        html += """
        </ul>
        
        <h3>Engagement Strategies</h3>
        <ul>
        """

        for strategy in section_data.get("engagement_strategies", []):
            html += f"<li>{strategy}</li>\n"

        html += """
        </ul>
        """

        return html

    def _create_report_sections(
        self, results: Dict[str, AnalysisResult]
    ) -> Dict[str, Any]:
        """
        Create sections for the report from analysis results.

        Args:
            results: Analysis results from different analyzers

        Returns:
            Dict[str, Any]: Report sections
        """
        sections = {}

        # Group results by analyzer category
        for analyzer_id, result in results.items():
            # Extract category from analyzer ID or use the analyzer_category property
            category = (
                result.analyzer_category
                if hasattr(result, "analyzer_category")
                else analyzer_id.split("_")[0]
            )

            # Convert result to dictionary
            result_dict = (
                result.model_dump()
                if hasattr(result, "model_dump")
                else (result.dict() if hasattr(result, "dict") else vars(result))
            )

            # Add to sections
            sections[category] = result_dict

        return sections

    def _create_report_summary(
        self, results: Dict[str, AnalysisResult], video_data: VideoData
    ) -> str:
        """
        Create a summary for the report.

        Args:
            results: Analysis results from different analyzers
            video_data: Video data that was analyzed

        Returns:
            str: Report summary
        """
        # Extract key metrics from results
        metrics = {}

        # Check for hook analysis
        hook_results = next(
            (r for r in results.values() if isinstance(r, HookAnalysisResult)), None
        )
        if hook_results:
            metrics["hook_effectiveness"] = hook_results.hook_effectiveness

        # Check for progression analysis
        progression_results = next(
            (r for r in results.values() if isinstance(r, ProgressionAnalysisResult)),
            None,
        )
        if progression_results:
            metrics["narrative_flow"] = progression_results.narrative_flow_score

        # Check for visual analysis
        visual_results = next(
            (r for r in results.values() if isinstance(r, VisualAnalysisResult)), None
        )
        if visual_results:
            metrics["visual_quality"] = visual_results.lighting_quality

        # Check for audio analysis
        audio_results = next(
            (r for r in results.values() if isinstance(r, AudioAnalysisResult)), None
        )
        if audio_results:
            metrics["audio_quality"] = audio_results.sound_quality

        # Create summary text
        video_name = video_data.path.name
        duration = video_data.duration
        resolution = f"{video_data.resolution[0]}x{video_data.resolution[1]}"

        summary = f"Analysis of video '{video_name}' ({duration:.1f}s, {resolution})"

        # Add metrics to summary if available
        if metrics:
            summary += " shows the following key metrics:\n"
            for name, value in metrics.items():
                summary += f"- {name.replace('_', ' ').title()}: {value * 100:.1f}%\n"
        else:
            summary += "."

        return summary

    def _collect_recommendations(self, results: Dict[str, AnalysisResult]) -> List[str]:
        """
        Collect recommendations from all analyzers.

        Args:
            results: Analysis results from different analyzers

        Returns:
            List[str]: Collected recommendations
        """
        recommendations = []

        # Collect recommendations from each analyzer result
        for result in results.values():
            # Check for hook recommendations
            if isinstance(result, HookAnalysisResult) and result.recommendations:
                recommendations.extend(result.recommendations)

            # Check for visual recommendations
            elif (
                isinstance(result, VisualAnalysisResult)
                and result.visual_recommendations
            ):
                recommendations.extend(result.visual_recommendations)

            # Check for other recommendation fields
            elif hasattr(result, "recommendations") and getattr(
                result, "recommendations"
            ):
                recommendations.extend(getattr(result, "recommendations"))

        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)

        return unique_recommendations

    def _extract_visual_examples(
        self, results: Dict[str, AnalysisResult], video_data: VideoData
    ) -> List[Dict[str, Any]]:
        """
        Extract visual examples from the video based on analysis results.

        Args:
            results: Analysis results from different analyzers
            video_data: Video data that was analyzed

        Returns:
            List[Dict[str, Any]]: Visual examples with timestamps, descriptions, and image data
        """
        visual_examples = []

        # Limit the number of examples
        max_examples = self.config.max_visual_examples
        examples_count = 0

        # Extract examples from hook analysis
        hook_results = next(
            (r for r in results.values() if isinstance(r, HookAnalysisResult)), None
        )
        if hook_results and examples_count < max_examples:
            # Add example from the hook section
            hook_frame = self._get_frame_at_time(
                video_data, hook_results.hook_start_time
            )
            if hook_frame:
                visual_examples.append(
                    {
                        "timestamp": hook_results.hook_start_time,
                        "description": "Video hook introduction",
                        "category": "Hook",
                        "image_data": self._frame_to_base64(hook_frame),
                    }
                )
                examples_count += 1

            # Add examples from key moments
            for moment in hook_results.key_moments[: max_examples - examples_count]:
                frame = self._get_frame_at_time(video_data, moment.get("timestamp", 0))
                if frame:
                    visual_examples.append(
                        {
                            "timestamp": moment.get("timestamp", 0),
                            "description": moment.get("description", "Key moment"),
                            "category": "Hook",
                            "image_data": self._frame_to_base64(frame),
                        }
                    )
                    examples_count += 1
                    if examples_count >= max_examples:
                        break

        # Extract examples from visual analysis
        visual_results = next(
            (r for r in results.values() if isinstance(r, VisualAnalysisResult)), None
        )
        if visual_results and examples_count < max_examples:
            # Add examples from color schemes
            for scheme in visual_results.color_schemes[: max_examples - examples_count]:
                frame = self._get_frame_at_time(video_data, scheme.get("timestamp", 0))
                if frame:
                    visual_examples.append(
                        {
                            "timestamp": scheme.get("timestamp", 0),
                            "description": f"Color scheme: {scheme.get('mood', '')}",
                            "category": "Visual",
                            "image_data": self._frame_to_base64(frame),
                        }
                    )
                    examples_count += 1
                    if examples_count >= max_examples:
                        break

        # Extract examples from object detection
        object_results = next(
            (r for r in results.values() if isinstance(r, ObjectDetectionResult)), None
        )
        if object_results and examples_count < max_examples:
            # Add examples from detected objects
            for obj in object_results.objects[: max_examples - examples_count]:
                frame = self._get_frame_at_time(video_data, obj.get("timestamp", 0))
                if frame:
                    visual_examples.append(
                        {
                            "timestamp": obj.get("timestamp", 0),
                            "description": f"Detected: {obj.get('label', '')}",
                            "category": "Object",
                            "image_data": self._frame_to_base64(frame),
                        }
                    )
                    examples_count += 1
                    if examples_count >= max_examples:
                        break

        # Extract examples from emotion analysis
        emotion_results = next(
            (r for r in results.values() if isinstance(r, EmotionAnalysisResult)), None
        )
        if emotion_results and examples_count < max_examples:
            # Add examples from emotional shifts
            for shift in emotion_results.emotional_shifts[
                : max_examples - examples_count
            ]:
                frame = self._get_frame_at_time(video_data, shift.get("timestamp", 0))
                if frame:
                    visual_examples.append(
                        {
                            "timestamp": shift.get("timestamp", 0),
                            "description": f"Emotional shift: {shift.get('from_emotion', '')} to {shift.get('to_emotion', '')}",
                            "category": "Emotion",
                            "image_data": self._frame_to_base64(frame),
                        }
                    )
                    examples_count += 1
                    if examples_count >= max_examples:
                        break

        return visual_examples

    def _get_frame_at_time(
        self, video_data: VideoData, timestamp: float
    ) -> Optional[Frame]:
        """
        Get the frame closest to the specified timestamp.

        Args:
            video_data: Video data
            timestamp: Timestamp in seconds

        Returns:
            Optional[Frame]: The closest frame or None if no frames are available
        """
        return video_data.get_frame_at_time(timestamp)

    def _frame_to_base64(self, frame: Frame) -> str:
        """
        Convert a frame to a base64-encoded JPEG string.

        Args:
            frame: Video frame

        Returns:
            str: Base64-encoded JPEG string
        """
        # Convert numpy array to PIL Image
        image = Image.fromarray(frame.image)

        # Resize if needed
        max_width, max_height = self.config.visual_example_max_size
        if image.width > max_width or image.height > max_height:
            image.thumbnail((max_width, max_height), Image.LANCZOS)

        # Convert to JPEG
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.config.visual_example_quality)

        # Convert to base64
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

"""
Command-line interface for the Video Analyzer.

This module provides a Typer-based CLI for the Video Analyzer, allowing users to:
- Analyze videos with various analyzers
- List available analyzers
- View examples of common usage patterns
- Generate reports in different formats

Usage examples:
    # Analyze a video with all available analyzers
    $ video-analyzer analyze my_video.mp4

    # Analyze a video with specific analyzers
    $ video-analyzer analyze my_video.mp4 --analyzers hook,progression,visual

    # Save analysis results to a file
    $ video-analyzer analyze my_video.mp4 --output-path report.json

    # List all available analyzers
    $ video-analyzer list-analyzers

    # Show examples of common usage patterns
    $ video-analyzer examples
"""

import asyncio
import json
import time
import sys
import signal
from pathlib import Path
from typing import Optional, List, Dict, Any
import typer
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from video_analyzer.config.analysis_pipeline import AnalysisPipelineConfig
from video_analyzer.config.video_processor import VideoProcessorConfig
from video_analyzer.config.frame_extractor import FrameExtractorConfig
from video_analyzer.services.analysis_manager import AnalysisManager
from video_analyzer.analyzers.base import AnalyzerRegistry, CancellationError
from video_analyzer.utils.errors import (
    VideoAnalyzerError,
    VideoProcessingError,
    AnalysisError,
    ReportGenerationError,
    ExternalServiceError,
)
from video_analyzer.utils.error_handling import (
    handle_errors,
    create_error_context,
    get_error_recovery_strategy,
)
from video_analyzer.utils.logging_config import configure_logging

# Configure logging
configure_logging(level="info")

# Get logger for this module
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Video Analyzer CLI - An AI-powered tool for comprehensive video analysis",
    no_args_is_help=True,
)
console = Console()


def generate_html_report(report):
    """
    Generate an HTML report from the analysis results.

    Args:
        report: The report object containing analysis results

    Returns:
        str: HTML content as a string
    """
    # Basic HTML template
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Analysis Report</title>
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
        .recommendation {{
            margin-bottom: 10px;
        }}
        .analyzer-section {{
            margin-bottom: 30px;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
        }}
        .analyzer-header {{
            display: flex;
            justify-content: space-between;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}
        .confidence {{
            background-color: #e8f4e8;
            padding: 5px 10px;
            border-radius: 15px;
            font-weight: bold;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 15px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
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
    </div>
    
    <div class="recommendations">
        <h2>Recommendations</h2>
        <ol>
"""

    # Add recommendations
    for recommendation in report.recommendations:
        html += f'            <li class="recommendation">{recommendation}</li>\n'

    html += """        </ol>
    </div>
    
    <h2>Detailed Analysis</h2>
"""

    # Add sections for each analyzer result
    for section_name, section_data in report.sections.items():
        html += f"""    <div class="analyzer-section">
        <div class="analyzer-header">
            <h3>{section_name}</h3>
"""

        # Add confidence if available
        if "confidence" in section_data:
            confidence = float(section_data["confidence"])
            html += f'            <span class="confidence">{confidence:.2f}</span>\n'

        html += """        </div>
"""

        # Add section content
        if isinstance(section_data, dict):
            html += """        <table>
            <tr>
                <th>Property</th>
                <th>Value</th>
            </tr>
"""

            for key, value in section_data.items():
                if key != "confidence" and not isinstance(value, (dict, list)):
                    html += f"            <tr>\n                <td>{key}</td>\n                <td>{value}</td>\n            </tr>\n"

            html += """        </table>
"""

        html += """    </div>
"""

    # Close the HTML
    html += """</body>
</html>
"""

    return html


@app.command(help="Analyze a video file and generate a detailed report")
def analyze(
    video_path: Path = typer.Argument(
        ...,
        help="Path to the video file to analyze",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    output_format: str = typer.Option(
        "json",
        "--format",
        "-f",
        help="Output format for the analysis report (json, html, pdf)",
        case_sensitive=False,
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save the output report",
        writable=True,
    ),
    parallel: bool = typer.Option(
        True,
        "--parallel/--sequential",
        help="Run analyzers in parallel or sequentially",
    ),
    analyzers: Optional[str] = typer.Option(
        None,
        "--analyzers",
        "-a",
        help="Specific analyzers to run (comma-separated list)",
    ),
    timeout: int = typer.Option(
        300,
        "--timeout",
        "-t",
        help="Timeout for each analyzer in seconds",
        min=30,
        max=3600,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output with detailed information",
    ),
):
    """
    Analyze a video file and generate a detailed report.

    This command processes the specified video file using various analyzers to extract
    insights about hooks, progression, visual elements, audio, objects, emotions, and
    storytelling techniques.

    Examples:
        $ video-analyzer analyze my_video.mp4
        $ video-analyzer analyze my_video.mp4 --analyzers hook,progression
        $ video-analyzer analyze my_video.mp4 --output report.json --format json
        $ video-analyzer analyze my_video.mp4 --sequential --timeout 600
    """
    # Validate the video path
    if not video_path.exists():
        console.print(f"[bold red]Error:[/bold red] Video file not found: {video_path}")
        raise typer.Exit(code=1)

    # Set up the configurations
    pipeline_config = AnalysisPipelineConfig(
        parallel_analyzers=parallel,
        timeout_seconds=timeout,
        enabled_analyzers=analyzers.split(",") if analyzers else None,
    )

    video_processor_config = VideoProcessorConfig()
    frame_extractor_config = FrameExtractorConfig()

    # Create the analysis manager
    manager = AnalysisManager(
        pipeline_config=pipeline_config,
        video_processor_config=video_processor_config,
        frame_extractor_config=frame_extractor_config,
    )

    # Register available analyzers
    available_analyzers = AnalyzerRegistry.get_available_types()
    console.print(f"Available analyzers: {', '.join(available_analyzers)}")

    analyzer_list = analyzers.split(",") if analyzers else available_analyzers

    for analyzer_type in analyzer_list:
        if analyzer_type in available_analyzers:
            try:
                manager.register_analyzer(analyzer_type)
                console.print(f"Registered analyzer: {analyzer_type}")
            except Exception as e:
                console.print(
                    f"[bold yellow]Warning:[/bold yellow] Failed to register analyzer {analyzer_type}: {str(e)}"
                )
        else:
            console.print(
                f"[bold yellow]Warning:[/bold yellow] Unknown analyzer type: {analyzer_type}"
            )

    # Set up progress tracking
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ) as progress:
        # Create a task for overall progress
        overall_task = progress.add_task("[bold]Overall progress", total=100)

        # Create tasks for each analyzer
        analyzer_tasks = {}
        for analyzer_id in manager.registered_analyzers:
            analyzer_tasks[analyzer_id] = progress.add_task(
                f"[cyan]{analyzer_id}", total=100, visible=False
            )

        # Add a task for the manager itself
        manager_task = progress.add_task(
            "[bold cyan]Processing", total=100, visible=True
        )

        # Set up progress callback
        def on_progress(
            analyzer_id: str, progress_value: float, metadata: Dict[str, Any]
        ) -> None:
            if analyzer_id == "pipeline":
                progress.update(overall_task, completed=int(progress_value * 100))
            elif analyzer_id == "manager":
                progress.update(manager_task, completed=int(progress_value * 100))

                # Update the task description based on status
                status = metadata.get("status", "in_progress")
                if status == "processing_video":
                    progress.update(
                        manager_task, description="[bold cyan]Processing video"
                    )
                elif status == "extracting_frames":
                    progress.update(
                        manager_task, description="[bold cyan]Extracting frames"
                    )
                elif status == "running_analysis":
                    progress.update(
                        manager_task, description="[bold cyan]Running analysis"
                    )
                elif status == "completed":
                    progress.update(
                        manager_task, description="[bold green]Processing completed"
                    )
                elif status == "error":
                    progress.update(
                        manager_task,
                        description=f"[bold red]Error: {metadata.get('error', 'unknown')}",
                    )
            elif analyzer_id in analyzer_tasks:
                # Make the task visible when it starts
                if (
                    progress_value > 0
                    and not progress.tasks[analyzer_tasks[analyzer_id]].visible
                ):
                    progress.update(analyzer_tasks[analyzer_id], visible=True)

                progress.update(
                    analyzer_tasks[analyzer_id], completed=int(progress_value * 100)
                )

                # Update the task description based on status
                status = metadata.get("status", "in_progress")
                if status == "completed":
                    progress.update(
                        analyzer_tasks[analyzer_id],
                        description=f"[green]{analyzer_id} (completed)",
                    )
                elif status == "error":
                    progress.update(
                        analyzer_tasks[analyzer_id],
                        description=f"[red]{analyzer_id} (error: {metadata.get('error', 'unknown')})",
                    )
                elif status == "cancelled":
                    progress.update(
                        analyzer_tasks[analyzer_id],
                        description=f"[yellow]{analyzer_id} (cancelled)",
                    )

        manager.set_progress_callback(on_progress)

        # Create a cancellation token
        cancellation_token = manager.create_cancellation_token()

        # Set up keyboard interrupt handling
        def handle_keyboard_interrupt():
            console.print("\n[bold yellow]Cancelling analysis...[/bold yellow]")
            cancellation_token.cancel()
            return True  # Signal that cancellation was handled

        # Process the video
        try:
            if verbose:
                console.print(
                    Panel(
                        f"Video: [cyan]{video_path}[/cyan]\n"
                        f"Format: [cyan]{output_format}[/cyan]\n"
                        f"Mode: [cyan]{'Parallel' if parallel else 'Sequential'}[/cyan]\n"
                        f"Analyzers: [cyan]{analyzers if analyzers else 'All'}[/cyan]\n"
                        f"Timeout: [cyan]{timeout}s[/cyan]",
                        title="Analysis Configuration",
                        border_style="blue",
                    )
                )
            else:
                console.print(f"[bold]Analyzing video:[/bold] {video_path}")

            console.print(
                "[yellow]Press Ctrl+C at any time to cancel the analysis[/yellow]"
            )

            # Log the analysis start
            logger.info(
                f"Starting video analysis: path={video_path}, format={output_format}, "
                f"parallel={parallel}, analyzers={analyzers if analyzers else 'All'}, "
                f"timeout={timeout}s"
            )

            # Run the analysis
            try:
                # Run the analysis in the event loop
                loop = asyncio.get_event_loop()
                results = loop.run_until_complete(manager.analyze_video(video_path))

                # Generate the report
                report = manager.generate_report(results, str(video_path))

                # Log successful analysis
                logger.info(
                    f"Analysis completed successfully: path={video_path}, "
                    f"analyzers_count={len(results)}"
                )

                console.print(
                    f"[bold green]Analysis completed with {len(results)} results[/bold green]"
                )

                # Save the results
                if output_path:
                    # Create parent directories if they don't exist
                    output_path.parent.mkdir(parents=True, exist_ok=True)

                    # Save the report based on the requested format
                    try:
                        if output_format.lower() == "json":
                            with open(output_path, "w") as f:
                                json.dump(report.dict(), f, indent=2)
                            logger.info(f"JSON report saved to {output_path}")
                            console.print(
                                f"[green]Report saved to {output_path}[/green]"
                            )
                        elif output_format.lower() == "html":
                            # Basic HTML report implementation
                            try:
                                html_content = generate_html_report(report)
                                with open(output_path, "w") as f:
                                    f.write(html_content)
                                logger.info(f"HTML report saved to {output_path}")
                                console.print(
                                    f"[green]HTML report saved to {output_path}[/green]"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to generate HTML report: {str(e)}. Saving as JSON instead.",
                                    exc_info=True,
                                )
                                console.print(
                                    f"[bold yellow]Warning:[/bold yellow] Failed to generate HTML report: {str(e)}. Saving as JSON instead."
                                )
                                json_path = output_path.with_suffix(".json")
                                with open(json_path, "w") as f:
                                    json.dump(report.dict(), f, indent=2)
                                logger.info(
                                    f"JSON report saved to {json_path} (fallback)"
                                )
                                console.print(
                                    f"[green]Report saved to {json_path}[/green]"
                                )
                        elif output_format.lower() == "pdf":
                            logger.warning(
                                "PDF output format is not yet fully supported, saving as JSON"
                            )
                            console.print(
                                f"[bold yellow]Warning:[/bold yellow] PDF output format is not yet fully supported, saving as JSON"
                            )
                            json_path = output_path.with_suffix(".json")
                            with open(json_path, "w") as f:
                                json.dump(report.dict(), f, indent=2)
                            logger.info(
                                f"JSON report saved to {json_path} (fallback from PDF)"
                            )
                            console.print(f"[green]Report saved to {json_path}[/green]")
                        else:
                            logger.warning(
                                f"Unknown output format {output_format}, saving as JSON"
                            )
                            console.print(
                                f"[bold yellow]Warning:[/bold yellow] Unknown output format {output_format}, saving as JSON"
                            )
                            json_path = output_path.with_suffix(".json")
                            with open(json_path, "w") as f:
                                json.dump(report.dict(), f, indent=2)
                            logger.info(
                                f"JSON report saved to {json_path} (fallback from unknown format)"
                            )
                            console.print(f"[green]Report saved to {json_path}[/green]")
                    except Exception as e:
                        # Handle file writing errors
                        error_context = create_error_context(
                            e,
                            {
                                "output_path": str(output_path),
                                "output_format": output_format,
                            },
                        )
                        logger.error(
                            f"Failed to save report: {str(e)}", extra=error_context
                        )
                        console.print(
                            f"[bold red]Error saving report:[/bold red] {str(e)}"
                        )

                        # Try to save to a default location as a fallback
                        try:
                            fallback_path = Path(
                                f"./video_analysis_report_{int(time.time())}.json"
                            )
                            with open(fallback_path, "w") as f:
                                json.dump(report.dict(), f, indent=2)
                            logger.info(
                                f"Report saved to fallback location: {fallback_path}"
                            )
                            console.print(
                                f"[green]Report saved to fallback location: {fallback_path}[/green]"
                            )
                        except Exception as fallback_error:
                            logger.error(
                                f"Failed to save report to fallback location: {str(fallback_error)}"
                            )
                            console.print(
                                "[bold red]Failed to save report to fallback location[/bold red]"
                            )
                else:
                    # Print a summary of the results
                    console.print("\n[bold]Analysis Results:[/bold]")
                    console.print(
                        Panel(
                            f"{report.summary}",
                            title="Summary",
                            border_style="green",
                        )
                    )

                    if report.recommendations:
                        recommendations_text = "\n".join(
                            [
                                f"{i}. {rec}"
                                for i, rec in enumerate(report.recommendations, 1)
                            ]
                        )
                        console.print(
                            Panel(
                                recommendations_text,
                                title="Recommendations",
                                border_style="cyan",
                            )
                        )

                    # Create a table for analyzer results
                    table = Table(title="Analyzer Results")
                    table.add_column("Analyzer", style="cyan")
                    table.add_column("Confidence", style="green")
                    table.add_column("Key Findings", style="yellow")

                    for analyzer_id, result in results.items():
                        # Extract a few key findings if available
                        key_findings = ""
                        if hasattr(result, "data") and isinstance(result.data, dict):
                            # Get up to 3 key findings
                            findings = []
                            count = 0
                            for k, v in result.data.items():
                                if count >= 3:
                                    break
                                if isinstance(v, (str, int, float, bool)):
                                    findings.append(f"{k}: {v}")
                                    count += 1
                            key_findings = "\n".join(findings)

                        table.add_row(
                            analyzer_id, f"{result.confidence:.2f}", key_findings
                        )

                    console.print(table)

                    # Notify user that analysis is complete
                    console.print(
                        "[bold green]Analysis complete![/bold green] Use --output to save the full report."
                    )

            except KeyboardInterrupt:
                if handle_keyboard_interrupt():
                    # Wait for cancellation to complete
                    try:
                        loop = asyncio.get_event_loop()
                        loop.run_until_complete(asyncio.sleep(1))
                    except:
                        pass
                    logger.info("Analysis cancelled by user")
                    console.print(
                        "[bold yellow]Analysis cancelled by user[/bold yellow]"
                    )
                    return

            except CancellationError:
                logger.info("Analysis was cancelled")
                console.print("[bold yellow]Analysis was cancelled[/bold yellow]")
                return

            except VideoProcessingError as e:
                # Handle video processing errors
                error_context = create_error_context(e, {"video_path": str(video_path)})
                logger.error(f"Video processing error: {str(e)}", extra=error_context)

                console.print(f"[bold red]Video processing error:[/bold red] {str(e)}")
                if verbose:
                    console.print("[red]Error details:[/red]")
                    if hasattr(e, "details") and e.details:
                        for key, value in e.details.items():
                            console.print(f"  [red]{key}:[/red] {value}")
                raise typer.Exit(code=1)

            except AnalysisError as e:
                # Handle analysis errors
                error_context = create_error_context(
                    e,
                    {
                        "video_path": str(video_path),
                        "analyzer_id": getattr(e, "analyzer_id", "unknown"),
                    },
                )
                logger.error(f"Analysis error: {str(e)}", extra=error_context)

                console.print(f"[bold red]Analysis error:[/bold red] {str(e)}")
                if verbose:
                    console.print("[red]Error details:[/red]")
                    if hasattr(e, "details") and e.details:
                        for key, value in e.details.items():
                            console.print(f"  [red]{key}:[/red] {value}")
                raise typer.Exit(code=1)

            except ReportGenerationError as e:
                # Handle report generation errors
                error_context = create_error_context(e, {"video_path": str(video_path)})
                logger.error(f"Report generation error: {str(e)}", extra=error_context)

                console.print(f"[bold red]Report generation error:[/bold red] {str(e)}")
                if verbose:
                    console.print("[red]Error details:[/red]")
                    if hasattr(e, "details") and e.details:
                        for key, value in e.details.items():
                            console.print(f"  [red]{key}:[/red] {value}")
                raise typer.Exit(code=1)

            except Exception as e:
                # Handle unexpected errors
                error_context = create_error_context(e, {"video_path": str(video_path)})
                logger.error(
                    f"Unexpected error during analysis: {str(e)}",
                    exc_info=True,
                    extra=error_context,
                )

                console.print(f"[bold red]Analysis failed:[/bold red] {str(e)}")
                if verbose:
                    import traceback

                    console.print("[red]Error details:[/red]")
                    console.print(traceback.format_exc())

                # Check if the error is potentially recoverable
                recovery_strategy = get_error_recovery_strategy(e)
                if recovery_strategy["recoverable"]:
                    console.print(
                        f"[yellow]Note:[/yellow] {recovery_strategy['message']}"
                    )

                raise typer.Exit(code=1)

        except VideoAnalyzerError as e:
            # Handle known application errors
            logger.error(f"Application error: {str(e)}", exc_info=True)
            console.print(f"[bold red]Error:[/bold red] {str(e)}")

            if verbose and hasattr(e, "to_dict"):
                error_dict = e.to_dict()
                console.print("[red]Error details:[/red]")
                for key, value in error_dict.items():
                    if key != "message" and key != "error_type":
                        console.print(f"  [red]{key}:[/red] {value}")

            raise typer.Exit(code=1)

        except Exception as e:
            # Handle truly unexpected errors
            logger.critical(f"Unexpected error: {str(e)}", exc_info=True)
            console.print(f"[bold red]Unexpected error:[/bold red] {str(e)}")

            if verbose:
                import traceback

                console.print("[red]Error details:[/red]")
                console.print(traceback.format_exc())

            raise typer.Exit(code=1)


@app.command(help="List all available video analyzers with descriptions")
def list_analyzers(
    detailed: bool = typer.Option(
        False,
        "--detailed",
        "-d",
        help="Show detailed information about each analyzer",
    ),
):
    """
    List all available analyzers with descriptions.

    This command displays information about all available analyzers that can be used
    with the analyze command. Use the --detailed flag to see more information about
    each analyzer's capabilities.

    Examples:
        $ video-analyzer list-analyzers
        $ video-analyzer list-analyzers --detailed
    """
    available_analyzers = AnalyzerRegistry.get_available_types()

    if detailed:
        # Create a rich table for detailed display
        table = Table(title="Available Analyzers", box=box.ROUNDED)
        table.add_column("Analyzer", style="cyan")
        table.add_column("Description")
        table.add_column("Capabilities", style="green")

        for analyzer_type in available_analyzers:
            try:
                analyzer_class = AnalyzerRegistry.get_analyzer_class(analyzer_type)
                # Create a temporary instance to get metadata
                analyzer = analyzer_class()
                capabilities = getattr(
                    analyzer, "capabilities", ["No detailed capabilities available"]
                )
                if isinstance(capabilities, list):
                    capabilities_str = "\n".join([f"• {cap}" for cap in capabilities])
                else:
                    capabilities_str = str(capabilities)

                table.add_row(analyzer_type, analyzer.description, capabilities_str)
            except Exception as e:
                table.add_row(
                    analyzer_type, f"[red]Error loading analyzer: {str(e)}[/red]", ""
                )

        console.print(table)
    else:
        # Simple list format
        console.print("[bold]Available Analyzers:[/bold]")

        for analyzer_type in available_analyzers:
            try:
                analyzer_class = AnalyzerRegistry.get_analyzer_class(analyzer_type)
                # Create a temporary instance to get metadata
                analyzer = analyzer_class()
                console.print(f"- [cyan]{analyzer_type}[/cyan]: {analyzer.description}")
            except Exception as e:
                console.print(
                    f"- [cyan]{analyzer_type}[/cyan]: [red]Error loading analyzer: {str(e)}[/red]"
                )


@app.command(help="Show examples of common usage patterns")
def examples():
    """
    Display examples of common usage patterns for the Video Analyzer CLI.

    This command shows various examples of how to use the Video Analyzer CLI
    for different analysis scenarios.
    """
    examples_panel = Panel(
        """
[bold]Basic Usage:[/bold]
  [cyan]$ video-analyzer analyze my_video.mp4[/cyan]
  Analyze a video with all available analyzers

[bold]Specific Analyzers:[/bold]
  [cyan]$ video-analyzer analyze my_video.mp4 --analyzers hook,progression,visual[/cyan]
  Analyze only specific aspects of a video

[bold]Output Options:[/bold]
  [cyan]$ video-analyzer analyze my_video.mp4 --output report.json[/cyan]
  Save analysis results to a file
  
  [cyan]$ video-analyzer analyze my_video.mp4 --format html --output report.html[/cyan]
  Generate an HTML report

[bold]Performance Options:[/bold]
  [cyan]$ video-analyzer analyze my_video.mp4 --sequential[/cyan]
  Run analyzers sequentially instead of in parallel
  
  [cyan]$ video-analyzer analyze my_video.mp4 --timeout 600[/cyan]
  Set a longer timeout for complex videos

[bold]Information Commands:[/bold]
  [cyan]$ video-analyzer list-analyzers[/cyan]
  List all available analyzers
  
  [cyan]$ video-analyzer list-analyzers --detailed[/cyan]
  Show detailed information about each analyzer
        """,
        title="Video Analyzer CLI Examples",
        border_style="green",
        expand=False,
    )

    console.print(examples_panel)


@app.command(help="Show version information")
def version():
    """
    Display version information for the Video Analyzer.
    """
    from importlib.metadata import version as pkg_version

    try:
        version = pkg_version("video_analyzer")
    except:
        version = "development"

    console.print(f"Video Analyzer version: [bold cyan]{version}[/bold cyan]")


@app.callback()
def main(
    ctx: typer.Context,
):
    """
    Video Analyzer - AI-powered comprehensive video analysis tool.

    This tool analyzes videos to provide insights about hooks, progression,
    visual elements, audio, objects, emotions, and storytelling techniques.
    """

    # Set up signal handling for graceful cancellation
    def signal_handler(sig, frame):
        console.print(
            "\n[bold yellow]Received interrupt signal. Cancelling...[/bold yellow]"
        )
        # The actual cancellation logic is handled in the analyze command
        # This just ensures we catch signals at the application level

    signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":
    app()

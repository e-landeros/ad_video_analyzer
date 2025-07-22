"""
Command-line interface for the Video Analyzer.
"""

import asyncio
import json
import time
import sys
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

from video_analyzer.config.analysis_pipeline import AnalysisPipelineConfig
from video_analyzer.config.video_processor import VideoProcessorConfig
from video_analyzer.config.frame_extractor import FrameExtractorConfig
from video_analyzer.services.analysis_manager import AnalysisManager
from video_analyzer.analyzers.base import AnalyzerRegistry, CancellationError
from video_analyzer.utils.errors import VideoAnalyzerError

app = typer.Typer(help="Video Analyzer CLI")
console = Console()


@app.command()
def analyze(
    video_path: Path = typer.Argument(..., help="Path to the video file"),
    output_format: str = typer.Option("json", help="Output format (json, html, pdf)"),
    output_path: Optional[Path] = typer.Option(None, help="Path to save the output"),
    parallel: bool = typer.Option(True, help="Run analyzers in parallel"),
    analyzers: Optional[str] = typer.Option(
        None, help="Analyzers to run (comma-separated)"
    ),
    timeout: int = typer.Option(300, help="Timeout for each analyzer in seconds"),
):
    """
    Analyze a video file and generate a detailed report.
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

        # Process the video
        try:
            console.print(f"[bold]Analyzing video:[/bold] {video_path}")

            # Run the analysis
            try:
                # Run the analysis in the event loop
                loop = asyncio.get_event_loop()
                results = loop.run_until_complete(manager.analyze_video(video_path))

                # Generate the report
                report = manager.generate_report(results, str(video_path))

                console.print(
                    f"[bold green]Analysis completed with {len(results)} results[/bold green]"
                )

                # Save the results
                if output_path:
                    # Save the report based on the requested format
                    if output_format.lower() == "json":
                        with open(output_path, "w") as f:
                            json.dump(report.dict(), f, indent=2)
                        console.print(f"Report saved to {output_path}")
                    else:
                        console.print(
                            f"[bold yellow]Warning:[/bold yellow] Output format {output_format} not yet supported, saving as JSON"
                        )
                        with open(output_path, "w") as f:
                            json.dump(report.dict(), f, indent=2)
                        console.print(f"Report saved to {output_path}")
                else:
                    # Print a summary of the results
                    console.print("\n[bold]Analysis Results:[/bold]")
                    console.print(f"Summary: {report.summary}")
                    console.print("\n[bold]Recommendations:[/bold]")
                    for i, recommendation in enumerate(report.recommendations, 1):
                        console.print(f"{i}. {recommendation}")

                    console.print("\n[bold]Analyzer Results:[/bold]")
                    for analyzer_id, result in results.items():
                        console.print(
                            f"- {analyzer_id}: {result.confidence:.2f} confidence"
                        )

            except KeyboardInterrupt:
                handle_keyboard_interrupt()
                # Wait for cancellation to complete
                try:
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(asyncio.sleep(1))
                except:
                    pass
                console.print("[bold yellow]Analysis cancelled by user[/bold yellow]")

            except CancellationError:
                console.print("[bold yellow]Analysis was cancelled[/bold yellow]")

            except Exception as e:
                console.print(f"[bold red]Analysis failed:[/bold red] {str(e)}")
                raise typer.Exit(code=1)

        except VideoAnalyzerError as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            raise typer.Exit(code=1)

        except Exception as e:
            console.print(f"[bold red]Unexpected error:[/bold red] {str(e)}")
            raise typer.Exit(code=1)


@app.command()
def list_analyzers():
    """
    List all available analyzers.
    """
    available_analyzers = AnalyzerRegistry.get_available_types()

    console.print("[bold]Available Analyzers:[/bold]")

    for analyzer_type in available_analyzers:
        try:
            analyzer_class = AnalyzerRegistry.get_analyzer_class(analyzer_type)
            # Create a temporary instance to get metadata
            analyzer = analyzer_class()
            console.print(f"- {analyzer_type}: {analyzer.description}")
        except Exception as e:
            console.print(
                f"- {analyzer_type}: [red]Error loading analyzer: {str(e)}[/red]"
            )


if __name__ == "__main__":
    app()


if __name__ == "__main__":
    app()

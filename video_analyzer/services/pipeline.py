"""
Analysis pipeline for orchestrating video analysis.
"""

import asyncio
from typing import List, Dict, Any, Optional
import time

from video_analyzer.analyzers.base import BaseAnalyzer
from video_analyzer.models.video import VideoData
from video_analyzer.models.analysis import AnalysisResult, Report
from video_analyzer.config import AnalysisPipelineConfig
from video_analyzer.utils import AnalysisError


class AnalysisPipeline:
    """
    Orchestrates the various analyzers.
    """

    def __init__(self, config: AnalysisPipelineConfig):
        self.config = config
        self.analyzers: List[BaseAnalyzer] = []

    def register_analyzer(self, analyzer: BaseAnalyzer) -> None:
        """
        Register an analyzer with the pipeline.

        Args:
            analyzer: The analyzer to register.
        """
        self.analyzers.append(analyzer)

    async def run_analysis(self, video_data: VideoData) -> Dict[str, AnalysisResult]:
        """
        Run all registered analyzers on the video data.

        Args:
            video_data: The video data to analyze.

        Returns:
            Dict[str, AnalysisResult]: A dictionary mapping analyzer IDs to their results.
        """
        start_time = time.time()
        results: Dict[str, AnalysisResult] = {}

        if self.config.parallel_analyzers:
            # Run analyzers in parallel
            tasks = []
            for analyzer in self.analyzers:
                tasks.append(self._run_analyzer_with_timeout(analyzer, video_data))

            analyzer_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for analyzer, result in zip(self.analyzers, analyzer_results):
                if isinstance(result, Exception):
                    # Log the error but continue with other analyzers
                    print(f"Error in analyzer {analyzer.analyzer_id}: {str(result)}")
                else:
                    results[analyzer.analyzer_id] = result
        else:
            # Run analyzers sequentially
            for analyzer in self.analyzers:
                try:
                    result = await self._run_analyzer_with_timeout(analyzer, video_data)
                    results[analyzer.analyzer_id] = result
                except Exception as e:
                    # Log the error but continue with other analyzers
                    print(f"Error in analyzer {analyzer.analyzer_id}: {str(e)}")

        elapsed_time = time.time() - start_time
        print(f"Analysis completed in {elapsed_time:.2f} seconds")

        return results

    async def _run_analyzer_with_timeout(
        self, analyzer: BaseAnalyzer, video_data: VideoData
    ) -> AnalysisResult:
        """
        Run an analyzer with a timeout.

        Args:
            analyzer: The analyzer to run.
            video_data: The video data to analyze.

        Returns:
            AnalysisResult: The analysis result.

        Raises:
            AnalysisError: If the analyzer times out or fails.
        """
        try:
            return await asyncio.wait_for(
                analyzer.analyze(video_data), timeout=self.config.timeout_seconds
            )
        except asyncio.TimeoutError:
            raise AnalysisError(
                f"Analyzer {analyzer.analyzer_id} timed out after {self.config.timeout_seconds} seconds"
            )
        except Exception as e:
            raise AnalysisError(f"Analyzer {analyzer.analyzer_id} failed: {str(e)}")

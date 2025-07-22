"""
Visual elements analyzer for video quality analysis.

This module provides functionality to analyze visual aspects of videos,
evaluating lighting, color schemes, camera movements, and visual effects.
"""

import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2
import asyncio

from video_analyzer.analyzers.base import BaseAnalyzer, AnalyzerRegistry
from video_analyzer.models.video import VideoData, Frame
from video_analyzer.models.analysis import VisualAnalysisResult

# Set up logging
logger = logging.getLogger(__name__)

# Common visual effect types in videos
VISUAL_EFFECT_TYPES = [
    "text",
    "color_grading",
    "vignette",
    "blur",
    "transition",
    "overlay",
    "split_screen",
    "picture_in_picture",
    "slow_motion",
    "time_lapse",
    "zoom",
    "filter",
]

# Common camera movement types
CAMERA_MOVEMENT_TYPES = [
    "pan_left",
    "pan_right",
    "tilt_up",
    "tilt_down",
    "zoom_in",
    "zoom_out",
    "dolly_in",
    "dolly_out",
    "tracking",
    "static",
    "handheld",
    "complex",
]


@AnalyzerRegistry.register("visual")
class VisualElementsAnalyzer(BaseAnalyzer):
    """
    Analyzes visual aspects of videos.

    This analyzer evaluates lighting techniques and quality, identifies color schemes
    and their emotional impact, detects camera movements and framing techniques,
    recognizes visual effects, and provides recommendations for visual improvements.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the visual elements analyzer.

        Args:
            config: Optional configuration for the analyzer
        """
        super().__init__(config)

        # Default visual analysis parameters
        self.lighting_threshold_low = self._config.get("lighting_threshold_low", 40)
        self.lighting_threshold_high = self._config.get("lighting_threshold_high", 220)
        self.color_sample_interval = self._config.get("color_sample_interval", 5)
        self.movement_detection_threshold = self._config.get(
            "movement_detection_threshold", 20.0
        )
        self.effect_detection_sensitivity = self._config.get(
            "effect_detection_sensitivity", 0.7
        )

    @property
    def analyzer_id(self) -> str:
        """
        Get the unique identifier for this analyzer.

        Returns:
            str: The analyzer ID
        """
        return "visual_elements_analyzer"

    @property
    def supports_progress(self) -> bool:
        """
        Check if this analyzer supports progress reporting.

        Returns:
            bool: True if progress reporting is supported
        """
        return True

    @property
    def required_frames(self) -> Dict[str, Any]:
        """
        Get the frame requirements for this analyzer.

        Returns:
            Dict[str, Any]: The frame requirements
        """
        return {
            "min_frames": 10,
            "frame_interval": 1.0,  # 1 frame per second is sufficient for visual analysis
            "specific_timestamps": None,
        }

    async def analyze(self, video_data: VideoData) -> VisualAnalysisResult:
        """
        Analyze the visual elements of the video.

        Args:
            video_data: The video data to analyze

        Returns:
            VisualAnalysisResult: The visual analysis result
        """
        logger.info(f"Analyzing visual elements for video: {video_data.path}")

        # Ensure we have enough frames to analyze
        if len(video_data.frames) < 5:
            logger.warning(f"Not enough frames for visual analysis: {video_data.path}")
            # Create a minimal result with default values
            return VisualAnalysisResult(
                analyzer_id=self.analyzer_id,
                lighting_quality=0.5,  # Default middle value
                video_id=str(video_data.path),
                confidence=0.5,  # Low confidence due to lack of frames
            )

        # Evaluate lighting quality and identify issues
        lighting_quality, lighting_issues = await self._evaluate_lighting(video_data)

        # Identify color schemes
        color_schemes = await self._identify_color_schemes(video_data)

        # Detect camera movements
        camera_movements = await self._detect_camera_movements(video_data)

        # Identify visual effects
        visual_effects = await self._identify_visual_effects(video_data)

        # Generate recommendations
        visual_recommendations = await self._generate_recommendations(
            video_data,
            lighting_quality,
            lighting_issues,
            color_schemes,
            camera_movements,
            visual_effects,
        )

        # Create the result
        result = VisualAnalysisResult(
            analyzer_id=self.analyzer_id,
            lighting_quality=lighting_quality,
            color_schemes=color_schemes,
            camera_movements=camera_movements,
            visual_effects=visual_effects,
            visual_recommendations=visual_recommendations,
            video_id=str(video_data.path),
            timestamps=[frame.timestamp for frame in video_data.frames],
            confidence=0.8,  # Reasonable confidence for visual analysis
        )

        logger.info(f"Visual elements analysis completed for video: {video_data.path}")
        return result

    async def _evaluate_lighting(
        self, video_data: VideoData
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Evaluate lighting techniques and quality.

        Args:
            video_data: The video data

        Returns:
            Tuple[float, List[Dict[str, Any]]]: Lighting quality score and identified issues
        """
        logger.debug("Evaluating lighting quality")
        lighting_issues = []
        frame_lighting_scores = []

        for frame in video_data.frames:
            # Convert to grayscale for lighting analysis
            if len(frame.image.shape) == 3:  # Color image
                gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
            else:  # Already grayscale
                gray = frame.image

            # Calculate histogram to analyze brightness distribution
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            # Calculate basic lighting metrics
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            min_brightness = np.min(gray)
            max_brightness = np.max(gray)

            # Calculate lighting score based on brightness distribution
            # Ideal: well-distributed histogram without too many pixels at extremes
            dark_pixels_ratio = np.sum(hist[: self.lighting_threshold_low]) / np.sum(
                hist
            )
            bright_pixels_ratio = np.sum(hist[self.lighting_threshold_high :]) / np.sum(
                hist
            )

            # Penalize if too many pixels are too dark or too bright
            lighting_score = 1.0 - (dark_pixels_ratio * 0.5 + bright_pixels_ratio * 0.5)

            # Adjust score based on standard deviation (too low = flat lighting, too high = harsh contrast)
            std_factor = min(std_brightness / 50.0, 1.0)  # Normalize std dev
            lighting_score *= (
                0.5 + std_factor * 0.5
            )  # Adjust score by up to 50% based on std dev

            # Ensure score is within bounds
            lighting_score = max(0.0, min(lighting_score, 1.0))
            frame_lighting_scores.append(lighting_score)

            # Identify lighting issues
            if dark_pixels_ratio > 0.4:  # More than 40% of pixels are dark
                lighting_issues.append(
                    {
                        "timestamp": frame.timestamp,
                        "issue": "underexposed",
                        "description": "Frame is underexposed (too dark)",
                        "severity": dark_pixels_ratio,
                    }
                )
            elif bright_pixels_ratio > 0.4:  # More than 40% of pixels are bright
                lighting_issues.append(
                    {
                        "timestamp": frame.timestamp,
                        "issue": "overexposed",
                        "description": "Frame is overexposed (too bright)",
                        "severity": bright_pixels_ratio,
                    }
                )
            elif std_brightness < 20:  # Low contrast
                lighting_issues.append(
                    {
                        "timestamp": frame.timestamp,
                        "issue": "flat_lighting",
                        "description": "Frame has flat lighting (low contrast)",
                        "severity": 1.0 - (std_brightness / 40.0),
                    }
                )
            elif std_brightness > 80:  # High contrast
                lighting_issues.append(
                    {
                        "timestamp": frame.timestamp,
                        "issue": "harsh_contrast",
                        "description": "Frame has harsh contrast (too much difference between light and dark areas)",
                        "severity": (std_brightness - 80) / 40.0,
                    }
                )

        # Calculate overall lighting quality as average of frame scores
        overall_lighting_quality = (
            np.mean(frame_lighting_scores) if frame_lighting_scores else 0.5
        )

        # Consolidate similar issues to avoid repetition
        consolidated_issues = []
        timestamps_covered = set()

        for issue in sorted(
            lighting_issues, key=lambda x: (x["issue"], x["timestamp"])
        ):
            # Skip if this timestamp is already covered by a similar issue
            if issue["timestamp"] in timestamps_covered:
                continue

            # Find similar issues within a 3-second window
            similar_issues = [
                i
                for i in lighting_issues
                if i["issue"] == issue["issue"]
                and abs(i["timestamp"] - issue["timestamp"]) < 3.0
            ]

            if similar_issues:
                # Create a consolidated issue
                avg_severity = np.mean([i["severity"] for i in similar_issues])
                timestamps = [i["timestamp"] for i in similar_issues]
                timestamps_covered.update(timestamps)

                consolidated_issues.append(
                    {
                        "timestamp": min(timestamps),
                        "issue": issue["issue"],
                        "description": issue["description"],
                        "severity": avg_severity,
                        "duration": max(timestamps) - min(timestamps)
                        if len(timestamps) > 1
                        else 0.0,
                    }
                )

        return overall_lighting_quality, consolidated_issues

    async def _identify_color_schemes(
        self, video_data: VideoData
    ) -> List[Dict[str, Any]]:
        """
        Identify color schemes and their emotional impact.

        Args:
            video_data: The video data

        Returns:
            List[Dict[str, Any]]: Identified color schemes
        """
        logger.debug("Identifying color schemes")
        color_schemes = []

        # Sample frames at regular intervals
        sample_frames = video_data.frames[:: self.color_sample_interval]
        if not sample_frames:
            sample_frames = video_data.frames  # Use all frames if sample is empty

        for frame in sample_frames:
            # Ensure the frame is in RGB format
            if len(frame.image.shape) == 3:  # Color image
                # OpenCV uses BGR, convert to RGB
                rgb_image = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)

                # Reshape the image for k-means clustering
                pixels = rgb_image.reshape((-1, 3)).astype(np.float32)

                # Use k-means to find dominant colors
                k = 5  # Number of dominant colors to extract
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    100,
                    0.2,
                )
                _, labels, centers = cv2.kmeans(
                    pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
                )

                # Convert centers to uint8
                centers = centers.astype(np.uint8)

                # Count occurrences of each label to determine color proportions
                unique_labels, counts = np.unique(labels, return_counts=True)
                sorted_indices = np.argsort(-counts)  # Sort by count (descending)

                # Get the dominant colors in order of prevalence
                dominant_colors = [centers[i].tolist() for i in sorted_indices]

                # Calculate average color
                average_color = (
                    np.mean(rgb_image, axis=(0, 1)).astype(np.uint8).tolist()
                )

                # Convert to HSV for better color analysis
                hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

                # Calculate average saturation and value
                avg_saturation = np.mean(hsv_image[:, :, 1]) / 255.0
                avg_value = np.mean(hsv_image[:, :, 2]) / 255.0

                # Determine color temperature (warm vs. cool)
                # Simple heuristic: more red/yellow = warm, more blue = cool
                r, g, b = average_color
                if r > b + 20:  # Red dominates blue
                    temperature = "warm"
                elif b > r + 20:  # Blue dominates red
                    temperature = "cool"
                else:
                    temperature = "neutral"

                # Determine mood based on saturation and value
                mood = self._determine_color_mood(
                    avg_saturation * 255, avg_value * 255, dominant_colors
                )

                # Create color scheme entry
                color_schemes.append(
                    {
                        "timestamp": frame.timestamp,
                        "colors": dominant_colors,
                        "average_color": average_color,
                        "temperature": temperature,
                        "mood": mood,
                        "mood_description": self._get_mood_description(mood),
                        "saturation": avg_saturation,
                        "brightness": avg_value,
                    }
                )

        return color_schemes

    def _determine_color_mood(
        self, saturation: float, value: float, colors: List[List[int]]
    ) -> str:
        """
        Determine the mood conveyed by a color scheme.

        Args:
            saturation: Average saturation (0-255)
            value: Average value/brightness (0-255)
            colors: List of dominant colors

        Returns:
            str: Color mood description
        """
        # Check for monochromatic scheme
        if len(colors) > 1:
            color_diversity = 0
            for i in range(len(colors)):
                for j in range(i + 1, len(colors)):
                    color_diversity += sum(
                        abs(c1 - c2) for c1, c2 in zip(colors[i], colors[j])
                    )

            # If colors are very similar, it's monochromatic
            if color_diversity < 100:  # Low diversity threshold
                return "monochromatic"

        # Determine mood based on saturation and value
        if saturation < 50:
            if value < 80:
                return "dark"
            elif value > 180:
                return "bright"
            else:
                return "muted"
        elif saturation > 150:
            return "vibrant"
        else:
            if value > 180:
                return "bright"
            elif value < 80:
                return "dark"
            else:
                return "neutral"

    def _get_mood_description(self, mood: str) -> str:
        """
        Get a description for a color mood.

        Args:
            mood: Color mood

        Returns:
            str: Description of the mood
        """
        descriptions = {
            "monochromatic": "Unified, focused, elegant",
            "dark": "Mysterious, serious, dramatic",
            "bright": "Energetic, optimistic, attention-grabbing",
            "muted": "Subtle, sophisticated, calming",
            "vibrant": "Dynamic, exciting, emotional",
            "neutral": "Balanced, natural, unobtrusive",
        }
        return descriptions.get(mood, "Balanced, natural")

    async def _detect_camera_movements(
        self, video_data: VideoData
    ) -> List[Dict[str, Any]]:
        """
        Detect camera movements and framing techniques.

        Args:
            video_data: The video data

        Returns:
            List[Dict[str, Any]]: Detected camera movements
        """
        logger.debug("Detecting camera movements")
        camera_movements = []

        # Need at least 2 frames to detect movement
        if len(video_data.frames) < 2:
            return camera_movements

        # Process consecutive frame pairs to detect motion
        prev_frame = None
        current_movement = None

        for frame in video_data.frames:
            # Convert to grayscale for optical flow
            if len(frame.image.shape) == 3:  # Color image
                gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
            else:  # Already grayscale
                gray = frame.image

            # Resize for faster processing if needed
            if gray.shape[0] > 300 or gray.shape[1] > 300:
                gray = cv2.resize(gray, (300, int(300 * gray.shape[0] / gray.shape[1])))

            if prev_frame is not None:
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )

                # Convert flow to polar coordinates (magnitude and angle)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                # Calculate average magnitude and angle
                avg_mag = np.mean(mag)

                # Skip if magnitude is below threshold (minimal movement)
                if avg_mag < self.movement_detection_threshold:
                    # If we were tracking a movement, end it
                    if current_movement:
                        movement_duration = (
                            frame.timestamp - current_movement["timestamp"]
                        )
                        if (
                            movement_duration >= 0.5
                        ):  # Only record movements that last at least 0.5 seconds
                            current_movement["duration"] = movement_duration
                            camera_movements.append(current_movement)
                        current_movement = None
                    prev_frame = gray
                    continue

                # Determine movement type based on flow direction
                angles = ang[mag > self.movement_detection_threshold]
                if len(angles) == 0:
                    movement_type = "static"
                else:
                    # Convert angles to degrees
                    angles_deg = np.rad2deg(angles)

                    # Count angles in different directions
                    left = np.sum((angles_deg > 135) & (angles_deg <= 225))
                    right = np.sum((angles_deg > 315) | (angles_deg <= 45))
                    up = np.sum((angles_deg > 45) & (angles_deg <= 135))
                    down = np.sum((angles_deg > 225) & (angles_deg <= 315))

                    # Determine dominant direction
                    counts = [left, right, up, down]
                    max_count = max(counts)
                    total_count = sum(counts)

                    if max_count < 0.5 * total_count:  # No clear dominant direction
                        movement_type = "complex"
                    else:
                        idx = counts.index(max_count)
                        if idx == 0:
                            movement_type = "pan_left"
                        elif idx == 1:
                            movement_type = "pan_right"
                        elif idx == 2:
                            movement_type = "tilt_up"
                        else:
                            movement_type = "tilt_down"

                # Check if this is a continuation of the current movement
                if current_movement and current_movement["type"] == movement_type:
                    # Update magnitude if it's larger
                    if avg_mag > current_movement["magnitude"]:
                        current_movement["magnitude"] = avg_mag
                else:
                    # If we were tracking a different movement, end it
                    if current_movement:
                        movement_duration = (
                            frame.timestamp - current_movement["timestamp"]
                        )
                        if (
                            movement_duration >= 0.5
                        ):  # Only record movements that last at least 0.5 seconds
                            current_movement["duration"] = movement_duration
                            camera_movements.append(current_movement)

                    # Start tracking a new movement
                    current_movement = {
                        "timestamp": frame.timestamp,
                        "type": movement_type,
                        "duration": 0.0,  # Will be updated when movement ends
                        "magnitude": avg_mag,
                        "purpose": self._determine_movement_purpose(movement_type),
                    }

            prev_frame = gray

        # Handle the last movement if it exists
        if current_movement and len(video_data.frames) > 0:
            movement_duration = (
                video_data.frames[-1].timestamp - current_movement["timestamp"]
            )
            if (
                movement_duration >= 0.5
            ):  # Only record movements that last at least 0.5 seconds
                current_movement["duration"] = movement_duration
                camera_movements.append(current_movement)

        return camera_movements

    def _determine_movement_purpose(self, movement_type: str) -> str:
        """
        Determine the purpose of a camera movement.

        Args:
            movement_type: Type of camera movement

        Returns:
            str: Purpose of the movement
        """
        purposes = {
            "pan_left": "Reveal new information or follow subject movement",
            "pan_right": "Reveal new information or follow subject movement",
            "tilt_up": "Reveal vertical space or create dramatic effect",
            "tilt_down": "Follow subject or transition to new scene",
            "zoom_in": "Focus attention or increase emotional intensity",
            "zoom_out": "Reveal context or decrease emotional intensity",
            "static": "Focus on composition or dialogue",
            "complex": "Create dynamic visual interest or follow complex action",
        }
        return purposes.get(movement_type, "Enhance visual storytelling")

    async def _identify_visual_effects(
        self, video_data: VideoData
    ) -> List[Dict[str, Any]]:
        """
        Identify visual effects and their purpose.

        Args:
            video_data: The video data

        Returns:
            List[Dict[str, Any]]: Identified visual effects
        """
        logger.debug("Identifying visual effects")
        visual_effects = []

        for i, frame in enumerate(video_data.frames):
            # Skip first and last frames for comparison
            if i == 0 or i == len(video_data.frames) - 1:
                continue

            # Get previous and next frames for comparison
            prev_frame = video_data.frames[i - 1].image
            next_frame = video_data.frames[i + 1].image

            # Check for text overlays using edge detection
            gray = (
                cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
                if len(frame.image.shape) == 3
                else frame.image
            )
            edges = cv2.Canny(gray, 100, 200)
            text_likelihood = np.sum(edges) / (gray.shape[0] * gray.shape[1])

            if text_likelihood > 0.05:  # Threshold determined empirically
                visual_effects.append(
                    {
                        "timestamp": frame.timestamp,
                        "type": "text",
                        "purpose": "Display information or dialogue",
                        "confidence": min(text_likelihood * 10, 1.0),
                    }
                )

            # Check for color grading by comparing color histograms
            if len(frame.image.shape) == 3:  # Color image
                # Calculate color histograms
                hist_frame = cv2.calcHist(
                    [frame.image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
                )
                hist_frame = cv2.normalize(hist_frame, hist_frame).flatten()

                hist_prev = cv2.calcHist(
                    [prev_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
                )
                hist_prev = cv2.normalize(hist_prev, hist_prev).flatten()

                # Compare histograms
                hist_diff = cv2.compareHist(hist_frame, hist_prev, cv2.HISTCMP_CORREL)

                # If histograms are significantly different but image structure is similar,
                # it might indicate color grading
                if hist_diff < 0.85:
                    # Check if structure is similar using grayscale correlation
                    gray_frame = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
                    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

                    # Calculate structural similarity
                    structure_similarity = cv2.compareHist(
                        cv2.calcHist([gray_frame], [0], None, [256], [0, 256]),
                        cv2.calcHist([gray_prev], [0], None, [256], [0, 256]),
                        cv2.HISTCMP_CORREL,
                    )

                    if structure_similarity > 0.9:  # High structural similarity
                        visual_effects.append(
                            {
                                "timestamp": frame.timestamp,
                                "type": "color_grading",
                                "purpose": "Create mood or visual style",
                                "confidence": 1.0 - hist_diff,
                            }
                        )

            # Check for vignette effect (darkening around edges)
            if len(frame.image.shape) == 3:  # Color image
                h, w = frame.image.shape[:2]
                center_region = frame.image[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
                edge_regions = [
                    frame.image[: h // 4, :],  # Top
                    frame.image[3 * h // 4 :, :],  # Bottom
                    frame.image[:, : w // 4],  # Left
                    frame.image[:, 3 * w // 4 :],  # Right
                ]

                center_brightness = np.mean(center_region)
                edge_brightness = np.mean([np.mean(region) for region in edge_regions])

                brightness_ratio = (
                    edge_brightness / center_brightness
                    if center_brightness > 0
                    else 1.0
                )

                if brightness_ratio < 0.8:  # Edges are significantly darker
                    visual_effects.append(
                        {
                            "timestamp": frame.timestamp,
                            "type": "vignette",
                            "purpose": "Direct attention to center or create mood",
                            "confidence": 1.0 - brightness_ratio,
                        }
                    )

            # Check for blur effect
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 100:  # Low variance indicates blur
                # Check if it's intentional by comparing with adjacent frames
                prev_var = cv2.Laplacian(
                    cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    if len(prev_frame.shape) == 3
                    else prev_frame,
                    cv2.CV_64F,
                ).var()

                next_var = cv2.Laplacian(
                    cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
                    if len(next_frame.shape) == 3
                    else next_frame,
                    cv2.CV_64F,
                ).var()

                # If current frame is blurrier than adjacent frames, it might be intentional
                if laplacian_var < prev_var * 0.7 and laplacian_var < next_var * 0.7:
                    visual_effects.append(
                        {
                            "timestamp": frame.timestamp,
                            "type": "blur",
                            "purpose": "Create depth of field or dreamy effect",
                            "confidence": 1.0 - (laplacian_var / 100),
                        }
                    )

        return visual_effects

    async def _generate_recommendations(
        self,
        video_data: VideoData,
        lighting_quality: float,
        lighting_issues: List[Dict[str, Any]],
        color_schemes: List[Dict[str, Any]],
        camera_movements: List[Dict[str, Any]],
        visual_effects: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Generate recommendations for visual improvements.

        Args:
            video_data: The video data
            lighting_quality: Overall lighting quality score
            lighting_issues: Identified lighting issues
            color_schemes: Identified color schemes
            camera_movements: Detected camera movements
            visual_effects: Identified visual effects

        Returns:
            List[str]: Recommendations for visual improvements
        """
        recommendations = []

        # Lighting recommendations
        if lighting_quality < 0.6:
            recommendations.append(
                "Improve overall lighting quality for better visual clarity"
            )

            # Specific lighting issues
            underexposed_issues = [
                i for i in lighting_issues if i["issue"] == "underexposed"
            ]
            overexposed_issues = [
                i for i in lighting_issues if i["issue"] == "overexposed"
            ]
            flat_lighting_issues = [
                i for i in lighting_issues if i["issue"] == "flat_lighting"
            ]

            if underexposed_issues:
                recommendations.append(
                    "Add more light to underexposed scenes to improve visibility"
                )
            if overexposed_issues:
                recommendations.append(
                    "Reduce exposure in bright scenes to preserve detail"
                )
            if flat_lighting_issues:
                recommendations.append(
                    "Add more contrast to flat-lit scenes for visual interest"
                )
        elif lighting_quality > 0.8:
            recommendations.append(
                "Excellent lighting quality. Consider maintaining this standard in future videos"
            )

        # Color scheme recommendations
        if color_schemes:
            # Check for consistency in color temperature
            temperatures = [scheme["temperature"] for scheme in color_schemes]
            if len(set(temperatures)) > 1:
                recommendations.append(
                    "Consider maintaining more consistent color grading throughout the video"
                )

            # Check for mood appropriateness
            moods = [scheme["mood"] for scheme in color_schemes]
            if (
                "muted" in moods and video_data.duration < 60
            ):  # Short video with muted colors
                recommendations.append(
                    "Consider using more vibrant colors to increase engagement in short-form content"
                )

        # Camera movement recommendations
        if camera_movements:
            # Check for excessive movement
            if (
                len(camera_movements) > len(video_data.frames) / 5
            ):  # More than one movement every 5 frames
                recommendations.append(
                    "Consider reducing camera movements to create a more stable viewing experience"
                )

            # Check for lack of movement in long videos
            if len(camera_movements) < 3 and video_data.duration > 60:
                recommendations.append(
                    "Consider adding more dynamic camera movements to maintain viewer interest"
                )

            # Check for consistency in movement types
            movement_types = [m["type"] for m in camera_movements]
            if len(set(movement_types)) == 1 and len(movement_types) > 3:
                recommendations.append(
                    "Vary camera movement types to create more visual interest"
                )

        # Visual effects recommendations
        if not visual_effects and video_data.duration > 30:
            recommendations.append(
                "Consider adding subtle visual effects to enhance production value"
            )

        # If no recommendations, add a general positive note
        if not recommendations:
            recommendations.append(
                "Consider experimenting with more dynamic visual techniques to enhance engagement"
            )

        return recommendations

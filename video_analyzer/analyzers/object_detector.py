"""
Object detector for video analysis.

This module provides functionality to detect and analyze objects, faces, expressions, and brands
in videos, tracking their screen time and positioning.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
import asyncio

from video_analyzer.analyzers.base import BaseAnalyzer, AnalyzerRegistry
from video_analyzer.models.video import VideoData, Frame
from video_analyzer.models.analysis import ObjectDetectionResult

# Set up logging
logger = logging.getLogger(__name__)

# Common object categories for detection
COMMON_OBJECT_CATEGORIES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# Common facial expressions
FACIAL_EXPRESSIONS = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "surprised",
    "fearful",
    "disgusted",
    "confused",
]

# Common brand categories
BRAND_CATEGORIES = [
    "technology",
    "fashion",
    "food",
    "beverage",
    "automotive",
    "entertainment",
    "sports",
    "health",
    "beauty",
    "finance",
    "retail",
    "other",
]


@AnalyzerRegistry.register("object")
class ObjectDetector(BaseAnalyzer):
    """
    Detects and analyzes objects, faces, expressions, and brands in videos.

    This analyzer uses computer vision techniques to identify objects, faces, and brands,
    track their screen time and positioning, and analyze their integration into the content.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the object detector.

        Args:
            config: Optional configuration for the analyzer
        """
        super().__init__(config)

        # Default object detection parameters
        self.detection_confidence_threshold = self._config.get(
            "detection_confidence_threshold", 0.5
        )
        self.face_detection_confidence = self._config.get(
            "face_detection_confidence", 0.7
        )
        self.brand_detection_confidence = self._config.get(
            "brand_detection_confidence", 0.6
        )
        self.tracking_interval = self._config.get("tracking_interval", 1.0)  # seconds
        self.sample_rate = self._config.get("sample_rate", 1)  # analyze every Nth frame

        # Initialize object detection model
        self.object_detector = None
        self.face_detector = None
        self.brand_detector = None

        # Initialize trackers
        self.object_trackers = {}
        self.face_trackers = {}
        self.brand_trackers = {}

    @property
    def analyzer_id(self) -> str:
        """
        Get the unique identifier for this analyzer.

        Returns:
            str: The analyzer ID
        """
        return "object_detector"

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
            "frame_interval": 0.5,  # Need more frequent frames for object tracking
            "specific_timestamps": None,
        }

    async def analyze(self, video_data: VideoData) -> ObjectDetectionResult:
        """
        Analyze the video for objects, faces, and brands.

        Args:
            video_data: The video data to analyze

        Returns:
            ObjectDetectionResult: The object detection result
        """
        logger.info(f"Analyzing objects and brands for video: {video_data.path}")

        # Ensure we have enough frames to analyze
        if len(video_data.frames) < 5:
            logger.warning(f"Not enough frames for object detection: {video_data.path}")
            # Create a minimal result with default values
            return ObjectDetectionResult(
                analyzer_id=self.analyzer_id,
                video_id=str(video_data.path),
                confidence=0.5,  # Low confidence due to lack of frames
            )

        # Initialize models if needed
        await self._initialize_models()

        # Detect objects, faces, and brands
        objects = await self._detect_objects(video_data)
        faces = await self._detect_faces(video_data)
        brands = await self._detect_brands(video_data)

        # Analyze screen time
        screen_time_analysis = await self._analyze_screen_time(
            objects, faces, brands, video_data.duration
        )

        # Calculate brand integration score
        brand_integration_score = await self._calculate_brand_integration_score(
            brands, video_data
        )

        # Create the result
        result = ObjectDetectionResult(
            analyzer_id=self.analyzer_id,
            objects=objects,
            faces=faces,
            brands=brands,
            screen_time_analysis=screen_time_analysis,
            brand_integration_score=brand_integration_score,
            video_id=str(video_data.path),
            timestamps=[frame.timestamp for frame in video_data.frames],
            confidence=0.8,  # Reasonable confidence for object detection
        )

        logger.info(f"Object detection completed for video: {video_data.path}")
        return result

    async def _initialize_models(self):
        """
        Initialize the object detection models.
        """
        logger.debug("Initializing object detection models")

        # For this implementation, we'll use OpenCV's DNN module with pre-trained models
        try:
            # Initialize object detector (YOLO or SSD would be used in a real implementation)
            # Here we're using a placeholder that will be replaced with actual model loading
            self.object_detector = True  # Placeholder

            # Initialize face detector (OpenCV's face detector or a deep learning model)
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            # Initialize brand detector (would be a custom model in a real implementation)
            self.brand_detector = True  # Placeholder

            logger.debug("Object detection models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize object detection models: {str(e)}")
            raise

    async def _detect_objects(self, video_data: VideoData) -> List[Dict[str, Any]]:
        """
        Detect objects in the video frames.

        Args:
            video_data: The video data

        Returns:
            List[Dict[str, Any]]: Detected objects with timestamps and positions
        """
        logger.debug("Detecting objects in video frames")
        detected_objects = []

        # Process frames at the specified sample rate
        for i, frame in enumerate(video_data.frames):
            if i % self.sample_rate != 0:
                continue

            # In a real implementation, we would use the object detector model here
            # For this implementation, we'll simulate object detection

            # Simulate detecting 1-3 objects in each frame
            num_objects = np.random.randint(1, 4)
            for _ in range(num_objects):
                # Generate a random object from common categories
                object_label = np.random.choice(COMMON_OBJECT_CATEGORIES)

                # Generate a random bounding box
                height, width = frame.image.shape[:2]
                x = np.random.randint(0, width - 100)
                y = np.random.randint(0, height - 100)
                w = np.random.randint(50, min(200, width - x))
                h = np.random.randint(50, min(200, height - y))

                # Generate a random confidence score
                confidence = np.random.uniform(self.detection_confidence_threshold, 1.0)

                # Add the detected object
                detected_objects.append(
                    {
                        "timestamp": frame.timestamp,
                        "label": object_label,
                        "confidence": float(confidence),
                        "bounding_box": [
                            int(x),
                            int(y),
                            int(w),
                            int(h),
                        ],  # [x, y, width, height]
                        "frame_index": frame.index,
                        "position": {
                            "center_x": float(x + w / 2)
                            / width,  # Normalized coordinates (0-1)
                            "center_y": float(y + h / 2) / height,
                            "size": float(w * h)
                            / (width * height),  # Relative size in frame
                        },
                    }
                )

        # Track objects across frames to establish continuity
        tracked_objects = await self._track_objects(detected_objects, video_data)

        return tracked_objects

    async def _detect_faces(self, video_data: VideoData) -> List[Dict[str, Any]]:
        """
        Detect faces and expressions in the video frames.

        Args:
            video_data: The video data

        Returns:
            List[Dict[str, Any]]: Detected faces with timestamps and expressions
        """
        logger.debug("Detecting faces in video frames")
        detected_faces = []

        # Process frames at the specified sample rate
        for i, frame in enumerate(video_data.frames):
            if i % self.sample_rate != 0:
                continue

            # Convert to grayscale for face detection
            if len(frame.image.shape) == 3:  # Color image
                gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
            else:  # Already grayscale
                gray = frame.image

            # Detect faces using OpenCV's face detector
            faces = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            # If no faces detected, continue to next frame
            if len(faces) == 0:
                continue

            # Process each detected face
            for x, y, w, h in faces:
                # In a real implementation, we would use a facial expression classifier here
                # For this implementation, we'll simulate expression detection
                expression = np.random.choice(FACIAL_EXPRESSIONS)
                confidence = np.random.uniform(self.face_detection_confidence, 1.0)

                # Add the detected face
                height, width = frame.image.shape[:2]
                detected_faces.append(
                    {
                        "timestamp": frame.timestamp,
                        "bounding_box": [
                            int(x),
                            int(y),
                            int(w),
                            int(h),
                        ],  # [x, y, width, height]
                        "expression": expression,
                        "confidence": float(confidence),
                        "frame_index": frame.index,
                        "position": {
                            "center_x": float(x + w / 2)
                            / width,  # Normalized coordinates (0-1)
                            "center_y": float(y + h / 2) / height,
                            "size": float(w * h)
                            / (width * height),  # Relative size in frame
                        },
                    }
                )

        # Track faces across frames to establish continuity
        tracked_faces = await self._track_faces(detected_faces, video_data)

        return tracked_faces

    async def _detect_brands(self, video_data: VideoData) -> List[Dict[str, Any]]:
        """
        Detect brand logos and products in the video frames.

        Args:
            video_data: The video data

        Returns:
            List[Dict[str, Any]]: Detected brands with timestamps and positions
        """
        logger.debug("Detecting brands in video frames")
        detected_brands = []

        # Process frames at the specified sample rate
        for i, frame in enumerate(video_data.frames):
            if i % self.sample_rate != 0:
                continue

            # In a real implementation, we would use a brand detection model here
            # For this implementation, we'll simulate brand detection

            # Simulate detecting 0-2 brands in each frame with 30% probability
            if np.random.random() < 0.3:
                num_brands = np.random.randint(1, 3)
                for _ in range(num_brands):
                    # Generate a random brand name
                    brand_names = [
                        "TechCorp",
                        "SportsBrand",
                        "FoodCo",
                        "AutoMaker",
                        "FashionStyle",
                    ]
                    brand_name = np.random.choice(brand_names)

                    # Generate a random brand category
                    brand_category = np.random.choice(BRAND_CATEGORIES)

                    # Generate a random bounding box
                    height, width = frame.image.shape[:2]
                    x = np.random.randint(0, width - 80)
                    y = np.random.randint(0, height - 80)
                    w = np.random.randint(40, min(150, width - x))
                    h = np.random.randint(40, min(150, height - y))

                    # Generate a random confidence score
                    confidence = np.random.uniform(self.brand_detection_confidence, 1.0)

                    # Add the detected brand
                    detected_brands.append(
                        {
                            "timestamp": frame.timestamp,
                            "name": brand_name,
                            "category": brand_category,
                            "bounding_box": [
                                int(x),
                                int(y),
                                int(w),
                                int(h),
                            ],  # [x, y, width, height]
                            "confidence": float(confidence),
                            "frame_index": frame.index,
                            "position": {
                                "center_x": float(x + w / 2)
                                / width,  # Normalized coordinates (0-1)
                                "center_y": float(y + h / 2) / height,
                                "size": float(w * h)
                                / (width * height),  # Relative size in frame
                            },
                            "prominence": float(w * h)
                            / (width * height)
                            * confidence,  # Prominence score
                        }
                    )

        # Track brands across frames to establish continuity
        tracked_brands = await self._track_brands(detected_brands, video_data)

        return tracked_brands

    async def _track_objects(
        self, detected_objects: List[Dict[str, Any]], video_data: VideoData
    ) -> List[Dict[str, Any]]:
        """
        Track objects across frames to establish continuity.

        Args:
            detected_objects: List of detected objects
            video_data: The video data

        Returns:
            List[Dict[str, Any]]: Tracked objects with additional tracking information
        """
        logger.debug("Tracking objects across frames")

        # Sort objects by timestamp
        sorted_objects = sorted(detected_objects, key=lambda x: x["timestamp"])

        # Group objects by label
        objects_by_label = {}
        for obj in sorted_objects:
            label = obj["label"]
            if label not in objects_by_label:
                objects_by_label[label] = []
            objects_by_label[label].append(obj)

        # Track each object type
        tracked_objects = []
        for label, objects in objects_by_label.items():
            # Assign tracking IDs to objects of the same label
            current_id = 0
            last_timestamp = -float("inf")
            last_position = None

            for obj in objects:
                current_position = (
                    obj["position"]["center_x"],
                    obj["position"]["center_y"],
                )

                # If this is the first object or there's a significant time gap, start a new track
                if (
                    last_position is None
                    or obj["timestamp"] - last_timestamp > self.tracking_interval * 2
                ):
                    current_id += 1
                    obj["tracking_id"] = f"{label}_{current_id}"
                    obj["track_start"] = obj["timestamp"]
                    obj["track_end"] = obj["timestamp"]
                else:
                    # Check if this is likely the same object based on position
                    distance = (
                        (current_position[0] - last_position[0]) ** 2
                        + (current_position[1] - last_position[1]) ** 2
                    ) ** 0.5

                    if distance < 0.2:  # Threshold for considering it the same object
                        obj["tracking_id"] = f"{label}_{current_id}"
                        obj["track_start"] = tracked_objects[-1]["track_start"]
                        obj["track_end"] = obj["timestamp"]
                        # Update the end time of the previous object with the same ID
                        for prev_obj in reversed(tracked_objects):
                            if prev_obj["tracking_id"] == obj["tracking_id"]:
                                prev_obj["track_end"] = obj["timestamp"]
                                break
                    else:
                        # This is a new object
                        current_id += 1
                        obj["tracking_id"] = f"{label}_{current_id}"
                        obj["track_start"] = obj["timestamp"]
                        obj["track_end"] = obj["timestamp"]

                last_timestamp = obj["timestamp"]
                last_position = current_position
                tracked_objects.append(obj)

        return tracked_objects

    async def _track_faces(
        self, detected_faces: List[Dict[str, Any]], video_data: VideoData
    ) -> List[Dict[str, Any]]:
        """
        Track faces across frames to establish continuity.

        Args:
            detected_faces: List of detected faces
            video_data: The video data

        Returns:
            List[Dict[str, Any]]: Tracked faces with additional tracking information
        """
        logger.debug("Tracking faces across frames")

        # Sort faces by timestamp
        sorted_faces = sorted(detected_faces, key=lambda x: x["timestamp"])

        # Track faces using a similar approach to object tracking
        tracked_faces = []
        current_id = 0
        last_timestamp = -float("inf")
        last_position = None

        for face in sorted_faces:
            current_position = (
                face["position"]["center_x"],
                face["position"]["center_y"],
            )

            # If this is the first face or there's a significant time gap, start a new track
            if (
                last_position is None
                or face["timestamp"] - last_timestamp > self.tracking_interval * 2
            ):
                current_id += 1
                face["face_id"] = f"face_{current_id}"
                face["track_start"] = face["timestamp"]
                face["track_end"] = face["timestamp"]
            else:
                # Check if this is likely the same face based on position
                distance = (
                    (current_position[0] - last_position[0]) ** 2
                    + (current_position[1] - last_position[1]) ** 2
                ) ** 0.5

                if distance < 0.15:  # Stricter threshold for faces
                    face["face_id"] = f"face_{current_id}"
                    face["track_start"] = tracked_faces[-1]["track_start"]
                    face["track_end"] = face["timestamp"]
                    # Update the end time of the previous face with the same ID
                    for prev_face in reversed(tracked_faces):
                        if prev_face["face_id"] == face["face_id"]:
                            prev_face["track_end"] = face["timestamp"]
                            break
                else:
                    # This is a new face
                    current_id += 1
                    face["face_id"] = f"face_{current_id}"
                    face["track_start"] = face["timestamp"]
                    face["track_end"] = face["timestamp"]

            last_timestamp = face["timestamp"]
            last_position = current_position
            tracked_faces.append(face)

        return tracked_faces

    async def _track_brands(
        self, detected_brands: List[Dict[str, Any]], video_data: VideoData
    ) -> List[Dict[str, Any]]:
        """
        Track brands across frames to establish continuity.

        Args:
            detected_brands: List of detected brands
            video_data: The video data

        Returns:
            List[Dict[str, Any]]: Tracked brands with additional tracking information
        """
        logger.debug("Tracking brands across frames")

        # Sort brands by timestamp
        sorted_brands = sorted(detected_brands, key=lambda x: x["timestamp"])

        # Group brands by name
        brands_by_name = {}
        for brand in sorted_brands:
            name = brand["name"]
            if name not in brands_by_name:
                brands_by_name[name] = []
            brands_by_name[name].append(brand)

        # Track each brand
        tracked_brands = []
        for name, brands in brands_by_name.items():
            # Assign tracking IDs to brands of the same name
            current_id = 0
            last_timestamp = -float("inf")
            last_position = None

            for brand in brands:
                current_position = (
                    brand["position"]["center_x"],
                    brand["position"]["center_y"],
                )

                # If this is the first brand or there's a significant time gap, start a new track
                if (
                    last_position is None
                    or brand["timestamp"] - last_timestamp > self.tracking_interval * 2
                ):
                    current_id += 1
                    brand["brand_instance_id"] = f"{name}_{current_id}"
                    brand["track_start"] = brand["timestamp"]
                    brand["track_end"] = brand["timestamp"]
                else:
                    # Check if this is likely the same brand instance based on position
                    distance = (
                        (current_position[0] - last_position[0]) ** 2
                        + (current_position[1] - last_position[1]) ** 2
                    ) ** 0.5

                    if (
                        distance < 0.2
                    ):  # Threshold for considering it the same brand instance
                        brand["brand_instance_id"] = f"{name}_{current_id}"
                        brand["track_start"] = tracked_brands[-1]["track_start"]
                        brand["track_end"] = brand["timestamp"]
                        # Update the end time of the previous brand with the same ID
                        for prev_brand in reversed(tracked_brands):
                            if (
                                prev_brand["brand_instance_id"]
                                == brand["brand_instance_id"]
                            ):
                                prev_brand["track_end"] = brand["timestamp"]
                                break
                    else:
                        # This is a new brand instance
                        current_id += 1
                        brand["brand_instance_id"] = f"{name}_{current_id}"
                        brand["track_start"] = brand["timestamp"]
                        brand["track_end"] = brand["timestamp"]

                last_timestamp = brand["timestamp"]
                last_position = current_position
                tracked_brands.append(brand)

        return tracked_brands

    async def _analyze_screen_time(
        self,
        objects: List[Dict[str, Any]],
        faces: List[Dict[str, Any]],
        brands: List[Dict[str, Any]],
        video_duration: float,
    ) -> Dict[str, Any]:
        """
        Analyze screen time for detected objects, faces, and brands.

        Args:
            objects: List of detected objects
            faces: List of detected faces
            brands: List of detected brands
            video_duration: Duration of the video in seconds

        Returns:
            Dict[str, Any]: Screen time analysis
        """
        logger.debug("Analyzing screen time")

        # Calculate screen time by object label
        screen_time_by_label = {}
        for obj in objects:
            label = obj["label"]
            if label not in screen_time_by_label:
                screen_time_by_label[label] = 0.0

            # Add the duration this object instance was tracked
            duration = obj["track_end"] - obj["track_start"]
            screen_time_by_label[label] += duration

        # Calculate screen time by face
        screen_time_by_face = {}
        for face in faces:
            face_id = face["face_id"]
            if face_id not in screen_time_by_face:
                screen_time_by_face[face_id] = 0.0

            # Add the duration this face was tracked
            duration = face["track_end"] - face["track_start"]
            screen_time_by_face[face_id] += duration

        # Calculate screen time by brand
        screen_time_by_brand = {}
        for brand in brands:
            name = brand["name"]
            if name not in screen_time_by_brand:
                screen_time_by_brand[name] = 0.0

            # Add the duration this brand was tracked
            duration = brand["track_end"] - brand["track_start"]
            screen_time_by_brand[name] += duration

        # Calculate percentages of video duration
        screen_time_percentage_by_label = {
            label: min(time / video_duration * 100, 100.0)
            for label, time in screen_time_by_label.items()
        }

        screen_time_percentage_by_brand = {
            name: min(time / video_duration * 100, 100.0)
            for name, time in screen_time_by_brand.items()
        }

        # Find most prominent objects and brands
        prominent_objects = sorted(
            [(label, time) for label, time in screen_time_by_label.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:5]  # Top 5 objects

        prominent_brands = sorted(
            [(name, time) for name, time in screen_time_by_brand.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:3]  # Top 3 brands

        # Create the screen time analysis result
        screen_time_analysis = {
            "by_label": screen_time_by_label,
            "by_brand": screen_time_by_brand,
            "percentage_by_label": screen_time_percentage_by_label,
            "percentage_by_brand": screen_time_percentage_by_brand,
            "prominent_objects": [
                {
                    "label": label,
                    "screen_time": time,
                    "percentage": screen_time_percentage_by_label[label],
                }
                for label, time in prominent_objects
            ],
            "prominent_brands": [
                {
                    "name": name,
                    "screen_time": time,
                    "percentage": screen_time_percentage_by_brand[name],
                }
                for name, time in prominent_brands
            ],
            "total_faces": len(screen_time_by_face),
            "face_time": sum(screen_time_by_face.values()),
            "face_percentage": min(
                sum(screen_time_by_face.values()) / video_duration * 100, 100.0
            ),
        }

        return screen_time_analysis

    async def _calculate_brand_integration_score(
        self, brands: List[Dict[str, Any]], video_data: VideoData
    ) -> float:
        """
        Calculate a score for how well brands are integrated into the content.

        Args:
            brands: List of detected brands
            video_data: The video data

        Returns:
            float: Brand integration score (0.0-1.0)
        """
        logger.debug("Calculating brand integration score")

        # If no brands detected, return 0
        if not brands:
            return 0.0

        # Calculate brand integration score based on several factors:
        # 1. Number of unique brands (fewer is better for focused integration)
        # 2. Screen time (more is better, but not overwhelming)
        # 3. Positioning (center of frame is better than edges)
        # 4. Size (not too small, not too large)
        # 5. Consistency of appearance

        # Count unique brands
        unique_brands = set(brand["name"] for brand in brands)
        num_unique_brands = len(unique_brands)

        # Calculate average screen time per brand
        brand_durations = {}
        for brand in brands:
            name = brand["name"]
            if name not in brand_durations:
                brand_durations[name] = 0.0
            brand_durations[name] += brand["track_end"] - brand["track_start"]

        avg_screen_time_percentage = (
            sum(brand_durations.values()) / video_data.duration / num_unique_brands
        )

        # Calculate average positioning (distance from center)
        avg_center_distance = 0.0
        for brand in brands:
            # Distance from center (0,0 is top-left, 0.5,0.5 is center)
            center_x = brand["position"]["center_x"]
            center_y = brand["position"]["center_y"]
            distance = ((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5
            avg_center_distance += distance

        if brands:
            avg_center_distance /= len(brands)

        # Calculate average size
        avg_size = (
            sum(brand["position"]["size"] for brand in brands) / len(brands)
            if brands
            else 0
        )

        # Calculate consistency (how regularly brands appear)
        timestamps = sorted([brand["timestamp"] for brand in brands])
        if len(timestamps) > 1:
            intervals = [
                timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)
            ]
            avg_interval = sum(intervals) / len(intervals)
            std_interval = np.std(intervals) if len(intervals) > 1 else 0
            consistency = (
                1.0 / (1.0 + std_interval / avg_interval) if avg_interval > 0 else 0.0
            )
        else:
            consistency = 0.0

        # Calculate the final score
        # Ideal: 1-3 unique brands, 5-15% screen time, centered positioning, moderate size, consistent appearance

        # Brand count score (peaks at 2-3 brands)
        brand_count_score = 1.0 - abs(min(num_unique_brands, 5) - 2.5) / 2.5

        # Screen time score (peaks at 10% of video duration)
        screen_time_score = 1.0 - abs(min(avg_screen_time_percentage, 0.3) - 0.1) / 0.1

        # Positioning score (higher for centered brands)
        positioning_score = 1.0 - min(avg_center_distance / 0.7, 1.0)

        # Size score (peaks at 5% of frame)
        size_score = 1.0 - abs(min(avg_size, 0.2) - 0.05) / 0.05

        # Combine scores with weights
        integration_score = (
            brand_count_score * 0.2
            + screen_time_score * 0.3
            + positioning_score * 0.2
            + size_score * 0.2
            + consistency * 0.1
        )

        # Ensure score is between 0 and 1
        integration_score = max(0.0, min(integration_score, 1.0))

        return integration_score

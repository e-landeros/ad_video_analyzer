"""
Script to create test videos for integration testing.

This script creates a set of test videos with different characteristics
that can be used for testing different aspects of the video analyzer.
"""

import os
import cv2
import numpy as np
from pathlib import Path


def create_test_video(
    output_path: Path,
    duration_seconds: int = 5,
    fps: int = 30,
    resolution: tuple = (640, 480),
    content_type: str = "basic",
):
    """
    Create a test video file for testing.

    Args:
        output_path: Path where the video will be saved
        duration_seconds: Duration of the video in seconds
        fps: Frames per second
        resolution: Video resolution (width, height)
        content_type: Type of content to generate (basic, color_shift, scene_change)
    """
    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, resolution)

    # Create frames
    total_frames = duration_seconds * fps

    for i in range(total_frames):
        # Create a frame based on content type
        if content_type == "basic":
            frame = create_basic_frame(i, total_frames, resolution)
        elif content_type == "color_shift":
            frame = create_color_shift_frame(i, total_frames, resolution)
        elif content_type == "scene_change":
            frame = create_scene_change_frame(i, total_frames, resolution)
        else:
            frame = create_basic_frame(i, total_frames, resolution)

        # Add frame number text
        cv2.putText(
            frame,
            f"Frame {i}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Write the frame
        out.write(frame)

    # Release the VideoWriter
    out.release()

    print(f"Created test video at {output_path}")


def create_basic_frame(frame_index: int, total_frames: int, resolution: tuple):
    """Create a basic frame with a moving rectangle."""
    frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)

    # Calculate position of the rectangle
    x = int((frame_index / total_frames) * (resolution[0] - 100))
    y = int(resolution[1] / 2 - 50)

    # Draw a rectangle
    cv2.rectangle(frame, (x, y), (x + 100, y + 100), (0, 255, 0), -1)

    return frame


def create_color_shift_frame(frame_index: int, total_frames: int, resolution: tuple):
    """Create a frame with color shifting over time."""
    frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)

    # Calculate color based on frame index
    r = int(255 * (np.sin(frame_index / total_frames * 6 * np.pi) + 1) / 2)
    g = int(255 * (np.sin(frame_index / total_frames * 4 * np.pi + np.pi / 2) + 1) / 2)
    b = int(255 * (np.sin(frame_index / total_frames * 2 * np.pi + np.pi) + 1) / 2)

    # Fill the frame with the color
    frame[:, :, 0] = b
    frame[:, :, 1] = g
    frame[:, :, 2] = r

    # Draw a rectangle in the center
    center_x = resolution[0] // 2
    center_y = resolution[1] // 2
    cv2.rectangle(
        frame,
        (center_x - 50, center_y - 50),
        (center_x + 50, center_y + 50),
        (255 - b, 255 - g, 255 - r),
        -1,
    )

    return frame


def create_scene_change_frame(frame_index: int, total_frames: int, resolution: tuple):
    """Create a frame with scene changes at regular intervals."""
    # Change scene every second (assuming 30 fps)
    scene_index = frame_index // 30 % 4

    frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)

    if scene_index == 0:
        # Scene 1: Moving rectangle
        x = int((frame_index % 30) / 30 * (resolution[0] - 100))
        y = int(resolution[1] / 2 - 50)
        cv2.rectangle(frame, (x, y), (x + 100, y + 100), (0, 255, 0), -1)

    elif scene_index == 1:
        # Scene 2: Expanding circle
        radius = int(((frame_index % 30) / 30) * 100) + 10
        cv2.circle(
            frame,
            (resolution[0] // 2, resolution[1] // 2),
            radius,
            (0, 0, 255),
            -1,
        )

    elif scene_index == 2:
        # Scene 3: Diagonal line pattern
        for i in range(0, resolution[0] + resolution[1], 20):
            offset = (frame_index % 30) * 2
            cv2.line(
                frame,
                (0, i - offset),
                (i - offset, 0),
                (255, 255, 0),
                2,
            )

    else:
        # Scene 4: Checkerboard pattern
        cell_size = 40
        offset = (frame_index % 30) % cell_size
        for y in range(0, resolution[1], cell_size):
            for x in range(0, resolution[0], cell_size):
                if ((x + y) // cell_size) % 2 == 0:
                    cv2.rectangle(
                        frame,
                        (x - offset, y - offset),
                        (x + cell_size - offset, y + cell_size - offset),
                        (0, 255, 255),
                        -1,
                    )

    # Add scene number text
    cv2.putText(
        frame,
        f"Scene {scene_index + 1}",
        (resolution[0] - 150, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    return frame


def create_test_videos():
    """Create a set of test videos for integration testing."""
    # Create test video directory
    test_video_dir = Path("tests/test_data")
    test_video_dir.mkdir(exist_ok=True)

    # Create a basic test video
    create_test_video(test_video_dir / "test_video.mp4")

    # Create videos with different durations
    create_test_video(test_video_dir / "short_video.mp4", duration_seconds=2)
    create_test_video(test_video_dir / "medium_video.mp4", duration_seconds=10)
    create_test_video(test_video_dir / "long_video.mp4", duration_seconds=30)

    # Create videos with different content types
    create_test_video(
        test_video_dir / "color_shift_video.mp4",
        content_type="color_shift",
        duration_seconds=10,
    )
    create_test_video(
        test_video_dir / "scene_change_video.mp4",
        content_type="scene_change",
        duration_seconds=10,
    )

    # Create videos with different resolutions
    create_test_video(
        test_video_dir / "low_res_video.mp4",
        resolution=(320, 240),
        duration_seconds=5,
    )
    create_test_video(
        test_video_dir / "high_res_video.mp4",
        resolution=(1280, 720),
        duration_seconds=5,
    )

    print("Test videos created successfully")


if __name__ == "__main__":
    create_test_videos()

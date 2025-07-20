from setuptools import setup, find_packages

setup(
    name="video_analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "typer",
        "pydantic",
        "numpy",
        "opencv-python",
        "openai",
    ],
    entry_points={
        "console_scripts": [
            "video-analyzer=video_analyzer:run",
        ],
    },
    author="Video Analyzer Team",
    author_email="example@example.com",
    description="AI-powered video content analysis tool",
    keywords="video, analysis, AI, content creation",
    python_requires=">=3.8",
)

from setuptools import setup, find_packages

setup(
    name="fast_asd",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy==1.23.5",
        "filterpy==1.4.5",
        "opencv-python==4.7.0.72",
        "scenedetect[opencv]",
        "lapx",
        "sortedcontainers",
        "supervision",
        "vidgear[core]",
        "imageio[ffmpeg]",
        "torch",
        "python_speech_features",
        "pandas",
        "ultralytics"
    ],
    python_requires=">=3.8",
)

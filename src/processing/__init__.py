"""
Processing module - detection, classification, and tracking.

Provides:
    VideoProcessor: Main orchestrator for video processing
    MotionDetector: GMM-based motion detection with shape/cohesiveness filters
    HailoClassifier: Hierarchical classification (family, genus, species)
    InsectTracker: Persistent tracking across videos
    
    Path topology functions for confirming insects:
    - analyze_path_topology
    - check_track_consistency
"""

from .processor import VideoProcessor
from .detector import (
    MotionDetector, 
    Detection,
    analyze_path_topology,
    check_track_consistency,
)
from .classifier import HailoClassifier, HierarchicalClassification
from .tracker import InsectTracker, Track

__all__ = [
    "VideoProcessor",
    "MotionDetector", 
    "Detection",
    "HailoClassifier",
    "HierarchicalClassification",
    "InsectTracker",
    "Track",
    "analyze_path_topology",
    "check_track_consistency",
]

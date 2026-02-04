"""
Persistent insect tracker.

Wraps InsectTracker and maintains tracking state across video chunks.
Tracks are continuous - an insect crossing video boundaries keeps its ID.
"""

import numpy as np
import uuid
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box with tracking metadata."""
    x: float
    y: float
    width: float
    height: float
    frame_id: int
    track_id: Optional[str] = None
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float, 
                  frame_id: int, track_id: Optional[str] = None):
        """Create from x1,y1,x2,y2 coordinates."""
        return cls(x1, y1, x2 - x1, y2 - y1, frame_id, track_id)


@dataclass 
class Track:
    """Track with history of detections."""
    track_id: str
    detections: List[Dict] = field(default_factory=list)
    classifications: List[Dict] = field(default_factory=list)
    first_frame: int = 0
    last_frame: int = 0
    is_active: bool = True


class InsectTracker:
    """
    Persistent insect tracker using Hungarian algorithm.
    
    Maintains tracking state across video boundaries for continuous
    tracking of insects across 1-minute video chunks.
    """
    
    def __init__(self, config: dict, image_height: int, image_width: int):
        """
        Initialize the tracker.
        
        Args:
            config: Tracking configuration from settings.yaml (required keys:
                    max_lost_frames, w_dist, w_area, cost_threshold)
            image_height: Frame height for normalization
            image_width: Frame width for normalization
        """
        self.config = config
        self.image_height = image_height
        self.image_width = image_width
        self.max_dist = np.sqrt(image_height**2 + image_width**2)
        
        # Tracking parameters from config (no defaults - must be provided)
        self.max_frames = config["max_lost_frames"]
        self.w_dist = config["w_dist"]
        self.w_area = config["w_area"]
        self.cost_threshold = config["cost_threshold"]
        
        # State
        self.current_tracks: List[BoundingBox] = []
        self.lost_tracks: Dict[str, Dict] = {}
        self.all_tracks: Dict[str, Track] = {}
        
        # Global frame counter (continuous across videos)
        self.global_frame_count = 0
        
        logger.info(f"Tracker: {image_width}x{image_height}, "
                   f"max_lost={self.max_frames}, threshold={self.cost_threshold}")
    
    def update(self, detections: List[Tuple], frame_id: int) -> List[str]:
        """
        Update tracking with new detections.
        
        Args:
            detections: List of (x1, y1, x2, y2) bounding boxes
            frame_id: Frame number within current video
            
        Returns:
            List of track IDs corresponding to each detection
        """
        global_frame = self.global_frame_count + frame_id
        
        if not detections:
            self._move_all_to_lost()
            self._age_lost_tracks()
            return []
        
        new_boxes = [
            BoundingBox.from_xyxy(*det[:4], global_frame)
            for det in detections
        ]
        
        if not self.current_tracks and not self.lost_tracks:
            track_ids = self._assign_new_ids(new_boxes)
            self.current_tracks = new_boxes
            return track_ids
        
        all_previous = self.current_tracks + [
            info['box'] for info in self.lost_tracks.values()
        ]
        
        if not all_previous:
            track_ids = self._assign_new_ids(new_boxes)
            self.current_tracks = new_boxes
            return track_ids
        
        cost_matrix, n_prev, n_curr = self._build_cost_matrix(all_previous, new_boxes)
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        
        track_ids = [None] * len(new_boxes)
        assigned = set()
        recovered = set()
        
        for i, j in zip(row_idx, col_idx):
            if i < n_prev and j < n_curr:
                cost = cost_matrix[i, j]
                if cost < self.cost_threshold:
                    prev_id = all_previous[i].track_id
                    new_boxes[j].track_id = prev_id
                    track_ids[j] = prev_id
                    assigned.add(j)
                    
                    if prev_id in self.lost_tracks:
                        recovered.add(prev_id)
        
        for tid in recovered:
            del self.lost_tracks[tid]
        
        for j in range(n_curr):
            if j not in assigned:
                new_id = self._generate_track_id()
                new_boxes[j].track_id = new_id
                track_ids[j] = new_id
                self._create_track(new_id, global_frame)
        
        matched_ids = {track_ids[j] for j in assigned if track_ids[j]}
        for track in self.current_tracks:
            if track.track_id not in matched_ids and track.track_id not in recovered:
                if track.track_id not in self.lost_tracks:
                    self.lost_tracks[track.track_id] = {
                        'box': track,
                        'frames_lost': 0
                    }
        
        self._age_lost_tracks()
        self.current_tracks = new_boxes
        
        return track_ids
    
    def add_detection(self, track_id: str, detection_data: Dict) -> None:
        """Add detection data to a track's history."""
        if track_id not in self.all_tracks:
            self._create_track(track_id, detection_data.get('frame_number', 0))
        
        track = self.all_tracks[track_id]
        track.detections.append(detection_data)
        track.last_frame = detection_data.get('frame_number', track.last_frame)
    
    def add_classification(self, track_id: str, classification_data: Dict) -> None:
        """Add classification result to a track."""
        if track_id in self.all_tracks:
            self.all_tracks[track_id].classifications.append(classification_data)
    
    def finalize_video(self, frames_in_video: int) -> None:
        """Called after processing a video chunk."""
        self.global_frame_count += frames_in_video
    
    def get_all_tracks(self) -> Dict[str, Track]:
        """Get all tracks (active and completed)."""
        return self.all_tracks
    
    def _build_cost_matrix(self, prev_boxes: List[BoundingBox], 
                           curr_boxes: List[BoundingBox]) -> Tuple[np.ndarray, int, int]:
        """Build cost matrix for Hungarian algorithm."""
        n_prev = len(prev_boxes)
        n_curr = len(curr_boxes)
        n = max(n_prev, n_curr)
        
        cost_matrix = np.ones((n, n)) * 999.0
        
        for i in range(n_prev):
            for j in range(n_curr):
                cost_matrix[i, j] = self._calculate_cost(prev_boxes[i], curr_boxes[j])
        
        return cost_matrix, n_prev, n_curr
    
    def _calculate_cost(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate matching cost between two boxes."""
        cx1, cy1 = box1.center()
        cx2, cy2 = box2.center()
        
        dist = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
        norm_dist = dist / self.max_dist
        
        min_area = min(box1.area, box2.area)
        max_area = max(box1.area, box2.area)
        area_cost = 1 - (min_area / max_area if max_area > 0 else 1.0)
        
        return (norm_dist * self.w_dist) + (area_cost * self.w_area)
    
    def _generate_track_id(self) -> str:
        """Generate unique track ID."""
        return str(uuid.uuid4())
    
    def _assign_new_ids(self, boxes: List[BoundingBox]) -> List[str]:
        """Assign new track IDs to all boxes."""
        track_ids = []
        for box in boxes:
            new_id = self._generate_track_id()
            box.track_id = new_id
            track_ids.append(new_id)
            self._create_track(new_id, box.frame_id)
        return track_ids
    
    def _create_track(self, track_id: str, first_frame: int) -> None:
        """Create a new track entry."""
        if track_id not in self.all_tracks:
            self.all_tracks[track_id] = Track(
                track_id=track_id,
                first_frame=first_frame,
                last_frame=first_frame
            )
    
    def _move_all_to_lost(self) -> None:
        """Move all current tracks to lost."""
        for track in self.current_tracks:
            if track.track_id not in self.lost_tracks:
                self.lost_tracks[track.track_id] = {
                    'box': track,
                    'frames_lost': 0
                }
        self.current_tracks = []
    
    def _age_lost_tracks(self) -> None:
        """Age lost tracks and remove old ones."""
        to_remove = []
        for track_id, info in self.lost_tracks.items():
            info['frames_lost'] += 1
            if info['frames_lost'] > self.max_frames:
                to_remove.append(track_id)
                if track_id in self.all_tracks:
                    self.all_tracks[track_id].is_active = False
        
        for tid in to_remove:
            del self.lost_tracks[tid]
    
    def get_stats(self) -> Dict:
        """Get current tracking statistics."""
        return {
            'active_tracks': len(self.current_tracks),
            'lost_tracks': len(self.lost_tracks),
            'total_tracks': len(self.all_tracks),
            'global_frame_count': self.global_frame_count
        }

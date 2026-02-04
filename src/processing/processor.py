"""
Video processor for insect detection pipeline.

Processes video files: detection, tracking, classification.
Maintains state across video chunks for continuous tracking.

Matches exact functionality of inference.py VideoInferenceProcessor.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

from .detector import (
    MotionDetector, Detection,
    analyze_path_topology, check_track_consistency
)
from .classifier import HailoClassifier, HierarchicalClassification
from .tracker import InsectTracker

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Processes video files for insect detection and classification.
    
    Pipeline (matches inference.py):
        1. Detection & Tracking: Process all frames, detect motion, build tracks
        2. Topology Analysis: Determine which tracks are confirmed insects
        3. Classification: Classify ONLY confirmed tracks (saves compute)
        4. Hierarchical Aggregation: Combine per-frame predictions
    """
    
    def __init__(self, config: dict):
        """
        Initialize the video processor.
        
        Args:
            config: Full configuration from settings.yaml
        """
        self.config = config
        self.detection_config = config.get("detection", {})
        self.classification_config = config.get("classification", {})
        self.tracking_config = config.get("tracking", {})
        
        # Components (lazy initialized)
        self._detector: Optional[MotionDetector] = None
        self._classifier: Optional[HailoClassifier] = None
        self._tracker: Optional[InsectTracker] = None
        
        # State (persists across videos for continuous tracking)
        self._initialized = False
        self._frame_size: Tuple[int, int] = (0, 0)
        
        # Track data (matches inference.py structure)
        self.all_detections: List[Dict] = []
        self.track_paths: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self.track_areas: Dict[str, List[float]] = defaultdict(list)
        
        # Global frame counter for continuous tracking
        self.global_frame_count = 0
        
        logger.info("VideoProcessor initialized")
    
    def _ensure_initialized(self, width: int, height: int) -> None:
        """Initialize components with frame dimensions."""
        if self._initialized and self._frame_size == (width, height):
            return
        
        self._frame_size = (width, height)
        
        # Initialize detector (reset for each video to clear background model)
        self._detector = MotionDetector(self.detection_config)
        
        # Initialize classifier (once)
        if self._classifier is None:
            self._classifier = HailoClassifier(self.classification_config)
        
        # Initialize tracker (persistent across videos)
        if self._tracker is None:
            self._tracker = InsectTracker(
                self.tracking_config,
                image_height=height,
                image_width=width
            )
        
        self._initialized = True
        logger.info(f"Components initialized for {width}x{height}")
    
    def process_video(self, video_path: Path) -> Dict:
        """
        Process a single video file.
        
        Pipeline:
            1. Phase 1: Detection & Tracking (all frames)
            2. Phase 2: Topology Analysis (determine confirmed tracks)
            3. Phase 3: Classification (confirmed tracks only)
            4. Phase 4: Hierarchical Aggregation
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dict with processing results
        """
        logger.info(f"Processing video: {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError(f"Could not read FPS from video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        
        logger.info(f"  {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")
        
        # Initialize components
        self._ensure_initialized(width, height)
        
        # Reset per-video state (but keep tracker for continuous tracking)
        video_detections = []
        
        # Parse timestamp from filename
        video_timestamp = self._parse_timestamp(video_path.stem)
        
        # =================================================================
        # PHASE 1: Detection & Tracking
        # =================================================================
        logger.info("Phase 1: Detection & Tracking")
        
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_time = frame_num / fps
            global_frame = self.global_frame_count + frame_num
            
            # Detect
            detections, fg_mask = self._detector.detect(frame, frame_num)
            
            # Track
            bboxes = [d.bbox for d in detections]
            track_ids = self._tracker.update(bboxes, global_frame)
            
            # Process each detection
            for i, (det, track_id) in enumerate(zip(detections, track_ids)):
                if track_id is None:
                    continue
                
                x1, y1, x2, y2 = det.bbox
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)
                
                # Track consistency check
                if self.track_paths[track_id]:
                    prev_pos = self.track_paths[track_id][-1]
                    prev_area = self.track_areas[track_id][-1] if self.track_areas[track_id] else area
                    
                    if not check_track_consistency(
                        prev_pos, (cx, cy), prev_area, area,
                        self.detection_config["max_frame_jump"]
                    ):
                        # Reset track path
                        self.track_paths[track_id] = []
                        self.track_areas[track_id] = []
                
                # Store path and area
                self.track_paths[track_id].append((cx, cy))
                self.track_areas[track_id].append(area)
                
                # Store detection data
                detection_data = {
                    'timestamp': datetime.now().isoformat(),
                    'frame_number': frame_num,
                    'global_frame': global_frame,
                    'frame_time_seconds': frame_time,
                    'track_id': track_id,
                    'bbox': [x1, y1, x2, y2],
                    'bbox_normalized': [
                        (x1 + x2) / (2 * width),
                        (y1 + y2) / (2 * height),
                        (x2 - x1) / width,
                        (y2 - y1) / height
                    ],
                    'area': area,
                }
                
                self.all_detections.append(detection_data)
                video_detections.append(detection_data)
            
            frame_num += 1
            
            if frame_num % 100 == 0:
                logger.info(f"  Processed {frame_num}/{total_frames} frames")
        
        cap.release()
        
        logger.info(f"  Phase 1 complete: {len(video_detections)} detections, "
                   f"{len(set(d['track_id'] for d in video_detections))} tracks")
        
        # =================================================================
        # PHASE 2: Topology Analysis
        # =================================================================
        logger.info("Phase 2: Topology Analysis")
        
        confirmed_track_ids, all_track_info = self._analyze_tracks(video_detections)
        
        logger.info(f"  {len(confirmed_track_ids)} confirmed / "
                   f"{len(all_track_info)} total tracks")
        
        # =================================================================
        # PHASE 3: Classification (confirmed tracks only)
        # =================================================================
        logger.info("Phase 3: Classification (confirmed tracks only)")
        
        track_classifications = {}
        if confirmed_track_ids:
            track_classifications = self._classify_confirmed_tracks(
                video_path, video_detections, confirmed_track_ids
            )
        
        # =================================================================
        # PHASE 4: Hierarchical Aggregation
        # =================================================================
        logger.info("Phase 4: Hierarchical Aggregation")
        
        results = self._hierarchical_aggregation(
            video_detections, confirmed_track_ids, track_classifications, all_track_info
        )
        
        # Update global frame counter for continuous tracking
        self.global_frame_count += frame_num
        
        # Build final output
        output = self._build_output(
            video_path=video_path,
            video_timestamp=video_timestamp,
            fps=fps,
            total_frames=frame_num,
            duration=duration,
            video_detections=video_detections,
            confirmed_track_ids=confirmed_track_ids,
            results=results,
            all_track_info=all_track_info
        )
        
        logger.info(f"  Complete: {len(results)} confirmed tracks")
        
        return output
    
    def _analyze_tracks(self, video_detections: List[Dict]) -> Tuple[Set[str], Dict]:
        """
        Analyze tracks to determine which pass topology analysis.
        
        Returns:
            Tuple of (confirmed_track_ids set, all_track_info dict)
        """
        # Group detections by track
        track_detections = defaultdict(list)
        for det in video_detections:
            if det['track_id']:
                track_detections[det['track_id']].append(det)
        
        confirmed_track_ids = set()
        all_track_info = {}
        
        for track_id, detections in track_detections.items():
            # Get path for this track
            path = self.track_paths.get(track_id, [])
            
            # Analyze topology
            passes_topology, topology_metrics = analyze_path_topology(
                path, self.detection_config
            )
            
            frame_times = [d['frame_time_seconds'] for d in detections]
            
            track_info = {
                'track_id': track_id,
                'num_detections': len(detections),
                'first_frame_time': min(frame_times),
                'last_frame_time': max(frame_times),
                'duration': max(frame_times) - min(frame_times),
                'passes_topology': passes_topology,
                **topology_metrics
            }
            all_track_info[track_id] = track_info
            
            if passes_topology:
                confirmed_track_ids.add(track_id)
                logger.info(f"  Track {track_id[:8]}: CONFIRMED "
                           f"({len(detections)} dets, {track_info['duration']:.1f}s)")
            else:
                logger.debug(f"  Track {track_id[:8]}: unconfirmed "
                            f"({len(detections)} dets)")
        
        return confirmed_track_ids, all_track_info
    
    def _classify_confirmed_tracks(self, video_path: Path, video_detections: List[Dict],
                                   confirmed_track_ids: Set[str]) -> Dict[str, List[HierarchicalClassification]]:
        """
        Classify only confirmed tracks by re-reading relevant frames.
        """
        # Group detections by frame for efficient seeking
        frames_to_classify = defaultdict(list)
        for det in video_detections:
            if det['track_id'] in confirmed_track_ids:
                frames_to_classify[det['frame_number']].append(det)
        
        if not frames_to_classify:
            return {}
        
        cap = cv2.VideoCapture(str(video_path))
        track_classifications = defaultdict(list)
        
        frame_numbers = sorted(frames_to_classify.keys())
        current_frame = 0
        classified_count = 0
        
        for target_frame in frame_numbers:
            # Seek to frame
            while current_frame < target_frame:
                cap.read()
                current_frame += 1
            
            ret, frame = cap.read()
            if not ret:
                break
            current_frame += 1
            
            # Classify each detection in this frame
            for det in frames_to_classify[target_frame]:
                x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                crop = frame[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue
                
                # Classify
                classification = self._classifier.classify(crop)
                track_classifications[det['track_id']].append(classification)
                
                # Update detection with classification
                det['family'] = classification.family
                det['genus'] = classification.genus
                det['species'] = classification.species
                det['family_confidence'] = classification.family_confidence
                det['genus_confidence'] = classification.genus_confidence
                det['species_confidence'] = classification.species_confidence
                det['family_probs'] = classification.family_probs
                det['genus_probs'] = classification.genus_probs
                det['species_probs'] = classification.species_probs
                
                classified_count += 1
        
        cap.release()
        logger.info(f"  Classified {classified_count} detections from "
                   f"{len(confirmed_track_ids)} tracks")
        
        return dict(track_classifications)
    
    def _hierarchical_aggregation(self, video_detections: List[Dict],
                                  confirmed_track_ids: Set[str],
                                  track_classifications: Dict[str, List[HierarchicalClassification]],
                                  all_track_info: Dict) -> List[Dict]:
        """
        Aggregate predictions for confirmed tracks using hierarchical selection.
        """
        results = []
        
        for track_id in confirmed_track_ids:
            if track_id not in track_classifications:
                continue
            
            classifications = track_classifications[track_id]
            if not classifications:
                continue
            
            # Hierarchical aggregation
            final_pred = self._classifier.hierarchical_aggregate(classifications)
            
            if not final_pred:
                continue
            
            # Get track info
            track_info = all_track_info.get(track_id, {})
            
            result = {
                'track_id': track_id,
                'num_detections': track_info.get('num_detections', len(classifications)),
                'first_frame_time': track_info.get('first_frame_time', 0),
                'last_frame_time': track_info.get('last_frame_time', 0),
                'duration': track_info.get('duration', 0),
                'final_family': final_pred['family'],
                'final_genus': final_pred['genus'],
                'final_species': final_pred['species'],
                'family_confidence': final_pred['family_confidence'],
                'genus_confidence': final_pred['genus_confidence'],
                'species_confidence': final_pred['species_confidence'],
                'passes_topology': True,
                **{k: v for k, v in track_info.items() 
                   if k in ['net_displacement', 'revisit_ratio', 
                           'progression_ratio', 'directional_variance']}
            }
            
            results.append(result)
            
            logger.info(f"  Track {track_id[:8]}: {final_pred['family']} / "
                       f"{final_pred['genus']} / {final_pred['species']} "
                       f"({final_pred['species_confidence']:.1%})")
        
        return results
    
    def _build_output(self, video_path: Path, video_timestamp: Optional[datetime],
                      fps: float, total_frames: int, duration: float,
                      video_detections: List[Dict], confirmed_track_ids: Set[str],
                      results: List[Dict], all_track_info: Dict) -> Dict:
        """Build final output structure."""
        
        # Build tracks with per-frame predictions
        tracks_data = []
        
        for result in results:
            track_id = result['track_id']
            
            # Get all detections for this track
            track_dets = [d for d in video_detections if d['track_id'] == track_id]
            
            # Build frames list
            frames = []
            for det in track_dets:
                frame_data = {
                    'frame_number': det['frame_number'],
                    'timestamp_seconds': det['frame_time_seconds'],
                    'bbox': det['bbox'],
                }
                
                # Add per-frame prediction if classified
                if 'species' in det:
                    frame_data['prediction'] = {
                        'family': det['family'],
                        'genus': det['genus'],
                        'species': det['species'],
                        'family_confidence': det['family_confidence'],
                        'genus_confidence': det['genus_confidence'],
                        'species_confidence': det['species_confidence'],
                    }
                
                frames.append(frame_data)
            
            track_data = {
                'track_id': track_id,
                'final_prediction': {
                    'family': result['final_family'],
                    'genus': result['final_genus'],
                    'species': result['final_species'],
                    'family_confidence': result['family_confidence'],
                    'genus_confidence': result['genus_confidence'],
                    'species_confidence': result['species_confidence'],
                },
                'num_detections': result['num_detections'],
                'first_seen_seconds': result['first_frame_time'],
                'last_seen_seconds': result['last_frame_time'],
                'duration_seconds': result['duration'],
                'topology_metrics': {
                    'net_displacement': result.get('net_displacement'),
                    'revisit_ratio': result.get('revisit_ratio'),
                    'progression_ratio': result.get('progression_ratio'),
                    'directional_variance': result.get('directional_variance'),
                },
                'frames': frames
            }
            
            tracks_data.append(track_data)
        
        return {
            'video_file': video_path.name,
            'video_timestamp': video_timestamp.isoformat() if video_timestamp else None,
            'processing_timestamp': datetime.now().isoformat(),
            'video_info': {
                'fps': fps,
                'total_frames': total_frames,
                'duration_seconds': duration
            },
            'summary': {
                'total_detections': len(video_detections),
                'total_tracks': len(all_track_info),
                'confirmed_tracks': len(confirmed_track_ids),
                'unconfirmed_tracks': len(all_track_info) - len(confirmed_track_ids),
            },
            'tracks': tracks_data,
            'global_frame_count': self.global_frame_count
        }
    
    def _parse_timestamp(self, filename_stem: str) -> Optional[datetime]:
        """Parse timestamp from video filename."""
        try:
            parts = filename_stem.split('_')
            if len(parts) >= 2:
                date_str = parts[-2]
                time_str = parts[-1]
                return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        except (ValueError, IndexError):
            pass
        return None
    
    def get_crops_for_track(self, video_path: Path, track_id: str) -> List[Tuple[int, np.ndarray]]:
        """Extract all crops for a specific track."""
        track_dets = [d for d in self.all_detections if d['track_id'] == track_id]
        
        if not track_dets:
            return []
        
        crops = []
        cap = cv2.VideoCapture(str(video_path))
        
        frame_to_bbox = {d['frame_number']: d['bbox'] for d in track_dets}
        frame_numbers = sorted(frame_to_bbox.keys())
        
        current_frame = 0
        for target_frame in frame_numbers:
            while current_frame < target_frame:
                cap.read()
                current_frame += 1
            
            ret, frame = cap.read()
            if not ret:
                break
            current_frame += 1
            
            x1, y1, x2, y2 = [int(v) for v in frame_to_bbox[target_frame]]
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append((target_frame, crop))
        
        cap.release()
        return crops
    
    def clear_video_detections(self) -> None:
        """Clear detections for current video (call after saving results)."""
        # Keep track_paths and track_areas for tracks that might continue
        # Only clear detections list
        self.all_detections = []

"""
Video processor for insect detection pipeline.

Uses BugSpot core for detection, tracking, crop extraction, and composites.
Adds Hailo classification and hierarchical aggregation on top.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

from bugspot import DetectionPipeline

from .classifier import HailoClassifier, HierarchicalClassification

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Processes video files for insect detection and classification.
    
    Pipeline:
        1-4. Detection, Tracking, Topology, Crops & Composites (BugSpot)
        5.   Classification (Hailo)
        6.   Hierarchical Aggregation
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.detection_config = config.get("detection", {})
        self.classification_config = config.get("classification", {})
        self.tracking_config = config.get("tracking", {})
        self.output_config = config.get("output", {})
        
        # Build bugspot config (merge detection + tracking params)
        bugspot_config = dict(self.detection_config)
        bugspot_config["max_lost_frames"] = self.tracking_config.get("max_lost_frames", 45)
        bugspot_config["tracker_w_dist"] = self.tracking_config.get("w_dist", 0.6)
        bugspot_config["tracker_w_area"] = self.tracking_config.get("w_area", 0.4)
        bugspot_config["tracker_cost_threshold"] = self.tracking_config.get("cost_threshold", 0.3)
        
        # Core pipeline (BugSpot)
        self._pipeline = DetectionPipeline(bugspot_config)
        
        # Classifier (lazy loaded)
        self._classifier: Optional[HailoClassifier] = None
        
        logger.info("VideoProcessor initialized")
    
    def process_video(self, video_path: Path) -> Dict:
        """
        Process a single video file.
        
        Returns:
            Dict with processing results
        """
        logger.info(f"Processing video: {video_path.name}")
        
        # Determine output dirs
        save_composites_dir = None
        if self.output_config.get("save_composites", True):
            results_dir = Path(self.output_config.get("results_dir", "output/results"))
            save_composites_dir = str(results_dir / f"{video_path.stem}_composites")
        
        save_crops_dir = None
        if self.output_config.get("save_crops", True):
            results_dir = Path(self.output_config.get("results_dir", "output/results"))
            crops_subdir = self.output_config.get("crops_subdir", "crops")
            save_crops_dir = str(results_dir / f"{video_path.stem}_{crops_subdir}")
        
        # =================================================================
        # PHASES 1-4: Detection, Tracking, Topology, Crops & Composites
        # =================================================================
        result = self._pipeline.process_video(
            str(video_path),
            extract_crops=True,
            render_composites=save_composites_dir is not None,
            save_crops_dir=save_crops_dir,
            save_composites_dir=save_composites_dir,
        )
        
        logger.info(f"  BugSpot: {len(result.confirmed_tracks)} confirmed / "
                    f"{len(result.track_paths)} total tracks")
        
        # =================================================================
        # PHASE 5: Classification (confirmed tracks only)
        # =================================================================
        logger.info("Phase 5: Classification (confirmed tracks only)")
        
        track_classifications: Dict[str, List[HierarchicalClassification]] = {}
        
        if result.confirmed_tracks:
            if self._classifier is None:
                self._classifier = HailoClassifier(self.classification_config)
            
            classified_count = 0
            for track_id, track in result.confirmed_tracks.items():
                classifications = []
                for frame_num, crop in track.crops:
                    classification = self._classifier.classify(crop)
                    classifications.append(classification)
                    
                    # Update detection data with classification
                    for det in result.all_detections:
                        if det["track_id"] == track_id and det["frame_number"] == frame_num:
                            det["family"] = classification.family
                            det["genus"] = classification.genus
                            det["species"] = classification.species
                            det["family_confidence"] = classification.family_confidence
                            det["genus_confidence"] = classification.genus_confidence
                            det["species_confidence"] = classification.species_confidence
                            det["family_probs"] = classification.family_probs
                            det["genus_probs"] = classification.genus_probs
                            det["species_probs"] = classification.species_probs
                            break
                    
                    classified_count += 1
                
                if classifications:
                    track_classifications[track_id] = classifications
            
            logger.info(f"  Classified {classified_count} detections from "
                       f"{len(track_classifications)} tracks")
        
        # =================================================================
        # PHASE 6: Hierarchical Aggregation
        # =================================================================
        logger.info("Phase 6: Hierarchical Aggregation")
        
        aggregated = self._hierarchical_aggregation(
            result, track_classifications
        )
        
        # Build output
        video_timestamp = self._parse_timestamp(video_path.stem)
        
        output = self._build_output(
            video_path=video_path,
            video_timestamp=video_timestamp,
            pipeline_result=result,
            aggregated=aggregated,
        )
        
        logger.info(f"  Complete: {len(aggregated)} confirmed tracks")
        return output
    
    def _hierarchical_aggregation(
        self,
        result,
        track_classifications: Dict[str, List[HierarchicalClassification]],
    ) -> List[Dict]:
        """Aggregate per-frame classifications using hierarchical selection."""
        results = []
        
        for track_id, track in result.confirmed_tracks.items():
            if track_id not in track_classifications:
                continue
            
            classifications = track_classifications[track_id]
            if not classifications:
                continue
            
            final_pred = self._classifier.hierarchical_aggregate(classifications)
            if not final_pred:
                continue
            
            entry = {
                "track_id": track_id,
                "num_detections": track.num_detections,
                "first_frame_time": track.first_frame_time,
                "last_frame_time": track.last_frame_time,
                "duration": track.duration,
                "final_family": final_pred["family"],
                "final_genus": final_pred["genus"],
                "final_species": final_pred["species"],
                "family_confidence": final_pred["family_confidence"],
                "genus_confidence": final_pred["genus_confidence"],
                "species_confidence": final_pred["species_confidence"],
                "passes_topology": True,
                **track.topology_metrics,
            }
            results.append(entry)
            
            logger.info(f"  Track {track_id[:8]}: {final_pred['family']} / "
                       f"{final_pred['genus']} / {final_pred['species']} "
                       f"({final_pred['species_confidence']:.1%})")
        
        return results
    
    def _build_output(self, video_path: Path, video_timestamp: Optional[datetime],
                      pipeline_result, aggregated: List[Dict]) -> Dict:
        """Build final JSON output structure."""
        tracks_data = []
        
        for entry in aggregated:
            track_id = entry["track_id"]
            
            # Per-frame data
            track_dets = [d for d in pipeline_result.all_detections if d["track_id"] == track_id]
            frames = []
            for det in track_dets:
                frame_data = {
                    "frame_number": det["frame_number"],
                    "timestamp_seconds": det["frame_time_seconds"],
                    "bbox": det["bbox"],
                }
                if "species" in det:
                    frame_data["prediction"] = {
                        "family": det["family"],
                        "genus": det["genus"],
                        "species": det["species"],
                        "family_confidence": det["family_confidence"],
                        "genus_confidence": det["genus_confidence"],
                        "species_confidence": det["species_confidence"],
                    }
                frames.append(frame_data)
            
            tracks_data.append({
                "track_id": track_id,
                "final_prediction": {
                    "family": entry["final_family"],
                    "genus": entry["final_genus"],
                    "species": entry["final_species"],
                    "family_confidence": entry["family_confidence"],
                    "genus_confidence": entry["genus_confidence"],
                    "species_confidence": entry["species_confidence"],
                },
                "num_detections": entry["num_detections"],
                "first_seen_seconds": entry["first_frame_time"],
                "last_seen_seconds": entry["last_frame_time"],
                "duration_seconds": entry["duration"],
                "topology_metrics": {
                    k: entry.get(k)
                    for k in ["net_displacement", "revisit_ratio",
                             "progression_ratio", "directional_variance"]
                },
                "frames": frames,
            })
        
        vi = pipeline_result.video_info
        return {
            "video_file": video_path.name,
            "video_timestamp": video_timestamp.isoformat() if video_timestamp else None,
            "processing_timestamp": datetime.now().isoformat(),
            "video_info": {
                "fps": vi["fps"],
                "total_frames": vi["total_frames"],
                "duration_seconds": vi["duration"],
            },
            "summary": {
                "total_detections": len(pipeline_result.all_detections),
                "total_tracks": len(pipeline_result.track_paths),
                "confirmed_tracks": len(pipeline_result.confirmed_tracks),
                "unconfirmed_tracks": len(pipeline_result.track_paths) - len(pipeline_result.confirmed_tracks),
            },
            "tracks": tracks_data,
        }
    
    def _parse_timestamp(self, filename_stem: str) -> Optional[datetime]:
        """Parse timestamp from video filename."""
        try:
            parts = filename_stem.split("_")
            if len(parts) >= 2:
                return datetime.strptime(f"{parts[-2]}_{parts[-1]}", "%Y%m%d_%H%M%S")
        except (ValueError, IndexError):
            pass
        return None
    
    def clear_video_detections(self) -> None:
        """Clear per-video state. Keeps tracker for continuous tracking."""
        self._pipeline.clear()

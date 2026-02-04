"""
Output writer for detection results.

Saves:
    - JSON results with tracks and per-frame predictions
    - Crop images for each detection
"""

import cv2
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ResultsWriter:
    """
    Writes detection results to JSON and saves crop images.
    
    Output JSON format matches inference.py structure with
    hierarchical predictions (family, genus, species).
    """
    
    def __init__(self, config: dict):
        """
        Initialize the results writer.
        
        Args:
            config: Output configuration from settings.yaml
        """
        self.config = config
        self.results_dir = Path(config.get("results_dir", "output/results"))
        self.save_crops = config.get("save_crops", True)
        self.crops_subdir = config.get("crops_subdir", "crops")
        
        # Ensure output directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ResultsWriter initialized: {self.results_dir}")
    
    def write_results(self, results: Dict, video_path: Path,
                      processor=None) -> Dict[str, Path]:
        """
        Write results for a processed video.
        
        Args:
            results: Processing results from VideoProcessor
            video_path: Path to the processed video
            processor: VideoProcessor instance for crop extraction
            
        Returns:
            Dict with output file paths
        """
        video_name = video_path.stem
        output_paths = {}
        
        # Write JSON results
        json_path = self._write_json(results, video_name)
        output_paths['json'] = json_path
        
        # Save crops if enabled
        if self.save_crops and processor and results.get('tracks'):
            crops_dir = self._save_crops(results, video_path, processor)
            if crops_dir:
                output_paths['crops_dir'] = crops_dir
        
        return output_paths
    
    def _write_json(self, results: Dict, video_name: str) -> Path:
        """
        Write results to JSON file.
        
        Output format (matches inference.py):
        {
            "video_file": "edge26_20260202_143000.mp4",
            "video_timestamp": "2026-02-02T14:30:00",
            "processing_timestamp": "2026-02-02T14:31:05",
            "video_info": {"fps": 30, "total_frames": 1800, "duration_seconds": 60},
            "summary": {
                "total_detections": 245,
                "total_tracks": 12,
                "confirmed_tracks": 3,
                "unconfirmed_tracks": 9
            },
            "tracks": [
                {
                    "track_id": "uuid",
                    "final_prediction": {
                        "family": "Apidae",
                        "genus": "Apis",
                        "species": "Apis mellifera",
                        "family_confidence": 0.92,
                        "genus_confidence": 0.89,
                        "species_confidence": 0.87
                    },
                    "num_detections": 48,
                    "first_seen_seconds": 2.5,
                    "last_seen_seconds": 45.8,
                    "duration_seconds": 43.3,
                    "topology_metrics": {
                        "net_displacement": 320.5,
                        "revisit_ratio": 0.12,
                        "progression_ratio": 0.85,
                        "directional_variance": 0.45
                    },
                    "frames": [
                        {
                            "frame_number": 75,
                            "timestamp_seconds": 2.5,
                            "bbox": [120, 340, 180, 400],
                            "prediction": {
                                "family": "Apidae",
                                "genus": "Apis",
                                "species": "Apis mellifera",
                                "family_confidence": 0.91,
                                "genus_confidence": 0.88,
                                "species_confidence": 0.85
                            }
                        },
                        ...
                    ]
                },
                ...
            ]
        }
        """
        json_path = self.results_dir / f"{video_name}_results.json"
        
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved: {json_path}")
        
        # Log summary
        summary = results.get('summary', {})
        logger.info(f"  Total tracks: {summary.get('total_tracks', 0)}, "
                   f"Confirmed: {summary.get('confirmed_tracks', 0)}")
        
        return json_path
    
    def _save_crops(self, results: Dict, video_path: Path,
                    processor) -> Optional[Path]:
        """
        Save crop images for each confirmed track.
        
        Directory structure:
            results_dir/
                video_name_crops/
                    track_id_short/
                        frame_000075.jpg
                        frame_000150.jpg
                        ...
        """
        video_name = video_path.stem
        crops_dir = self.results_dir / f"{video_name}_{self.crops_subdir}"
        
        tracks = results.get('tracks', [])
        if not tracks:
            return None
        
        crops_saved = 0
        
        for track in tracks:
            track_id = track['track_id']
            track_short = track_id[:8]
            
            # Get crops from processor
            crops = processor.get_crops_for_track(video_path, track_id)
            
            if not crops:
                continue
            
            # Create track directory
            track_dir = crops_dir / track_short
            track_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each crop
            for frame_num, crop in crops:
                crop_path = track_dir / f"frame_{frame_num:06d}.jpg"
                cv2.imwrite(str(crop_path), crop)
                crops_saved += 1
        
        if crops_saved > 0:
            logger.info(f"Saved {crops_saved} crops to {crops_dir}")
            return crops_dir
        
        return None
    
    def write_summary(self, all_results: List[Dict]) -> Path:
        """
        Write a summary of all processed videos.
        
        Args:
            all_results: List of results from multiple videos
            
        Returns:
            Path to summary JSON
        """
        total_confirmed = 0
        total_tracks = 0
        species_counts = {}
        
        for result in all_results:
            summary = result.get('summary', {})
            total_confirmed += summary.get('confirmed_tracks', 0)
            total_tracks += summary.get('total_tracks', 0)
            
            for track in result.get('tracks', []):
                final_pred = track.get('final_prediction', {})
                species = final_pred.get('species', 'Unknown')
                species_counts[species] = species_counts.get(species, 0) + 1
        
        summary_data = {
            'generated_at': datetime.now().isoformat(),
            'total_videos': len(all_results),
            'total_tracks': total_tracks,
            'total_confirmed': total_confirmed,
            'species_counts': species_counts,
            'videos': [
                {
                    'video_file': r.get('video_file'),
                    'video_timestamp': r.get('video_timestamp'),
                    'confirmed_tracks': r.get('summary', {}).get('confirmed_tracks', 0),
                    'species_detected': [
                        t.get('final_prediction', {}).get('species')
                        for t in r.get('tracks', [])
                    ]
                }
                for r in all_results
            ]
        }
        
        summary_path = self.results_dir / 'processing_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"Summary saved: {summary_path}")
        return summary_path

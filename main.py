"""
Edge26 - On-Device Insect Detection Pipeline

Usage:
    python main.py --config config/settings.yaml
"""

import argparse
import logging
import os
import queue
import shutil
import signal
import sys
import threading
import yaml
from pathlib import Path
from datetime import datetime

from src.capture import VideoRecorder
from src.processing import VideoProcessor
from src.output import ResultsWriter


def setup_logging(log_dir: Path) -> None:
    """Configure logging to console and file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"edge26_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Format
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%H:%M:%S"
    
    # Root logger
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    
    # Reduce noise from libraries
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("hailo_platform").setLevel(logging.WARNING)


logger = logging.getLogger("edge26")


class Pipeline:
    """Main pipeline orchestrating capture and processing."""
    
    def __init__(self, config: dict):
        self.config = config
        self.video_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.recording_stopped = threading.Event()
        self.recorder_thread = None
        self.processor_thread = None
        
        # Device config
        device_config = config.get("device", {})
        self.flick_id = device_config.get("flick_id", "edge26")
        self.dot_ids = device_config.get("dot_ids", [])
        self.input_storage = Path(config["paths"]["input_storage"])
        
        # Pipeline mode
        pipeline_config = config.get("pipeline", {})
        self.enable_recording = pipeline_config.get("enable_recording", True)
        self.enable_processing = pipeline_config.get("enable_processing", True)
        self.enable_classification = pipeline_config.get("enable_classification", True)
        self.continuous_tracking = pipeline_config.get("continuous_tracking", True)
        
        # --- Tracker reset signals (continuous_tracking mode) ---
        # 1. Day-change: reset when the date in the filename changes
        self._last_video_date: str = ""
        # 2. Recording-stop: reset after the last recorded video is processed
        #    Persisted via .last_recording marker file so it survives restarts.
        self._reset_after_video: str = ""
        self._pending_tracker_reset = False
        if self.continuous_tracking:
            self._load_last_recording_marker()
        
        # Initialize components based on mode
        self.recorder = self._init_recorder() if self.enable_recording else None
        self.processor = VideoProcessor(config) if self.enable_processing else None
        self.writer = ResultsWriter(config["output"]) if self.enable_processing else None
        
        logger.info("=" * 60)
        logger.info("EDGE26 PIPELINE INITIALIZED")
        logger.info("=" * 60)
        
        # Mode info
        mode = "RECORD + PROCESS" if (self.enable_recording and self.enable_processing) else \
               "RECORD ONLY" if self.enable_recording else \
               "PROCESS ONLY" if self.enable_processing else "NONE"
        logger.info(f"Mode:          {mode}")
        logger.info(f"Device:        {self.flick_id}")
        logger.info(f"Input storage: {config['paths']['input_storage']}")
        if self.enable_processing:
            logger.info(f"Results dir:   {config['output']['results_dir']}")
            classify = pipeline_config.get("enable_classification", True)
            logger.info(f"Classification: {'enabled' if classify else 'disabled (detection only)'}")
            cont_track = pipeline_config.get("continuous_tracking", True)
            logger.info(f"Tracking:      {'continuous (across videos)' if cont_track else 'per-video (reset each)'}")
        if self.dot_ids:
            logger.info(f"DOT devices:   {', '.join(self.dot_ids)}")
        if self.enable_recording:
            rec_mode = pipeline_config.get("recording_mode", "continuous")
            logger.info(f"Chunk duration: {config['capture']['chunk_duration_seconds']}s")
            logger.info(f"Recording mode: {rec_mode}"
                       + (f" (every {pipeline_config.get('recording_interval_minutes', 5)} min)"
                          if rec_mode == "interval" else ""))
    
    def _init_recorder(self) -> VideoRecorder:
        """Initialize video recorder from config."""
        paths = self.config["paths"]
        capture = self.config["capture"]
        pipeline_cfg = self.config.get("pipeline", {})
        
        return VideoRecorder(
            output_dir=paths["input_storage"],
            fps=capture["fps"],
            chunk_duration=capture["chunk_duration_seconds"],
            device_id=self.flick_id,
            video_queue=self.video_queue,
            camera_index=capture["camera_index"],
            use_picamera=capture["use_picamera"],
            recording_mode=pipeline_cfg.get("recording_mode", "continuous"),
            interval_minutes=pipeline_cfg.get("recording_interval_minutes", 5),
        )
    
    def _is_dot_directory(self, path: Path) -> bool:
        """Check if a path is a DOT device directory."""
        if not path.is_dir():
            return False
        return any(path.name.startswith(f"{dot_id}_") for dot_id in self.dot_ids)
    
    def _find_existing_items(self) -> list:
        """
        Find existing videos and DOT directories in input_storage.
        
        Returns a sorted list of (path, type) tuples where type is
        "video" or "dot". Sorted by name gives chronological order
        since filenames and directory names both contain timestamps.
        """
        if not self.input_storage.exists():
            return []
        
        items = []
        for entry in sorted(self.input_storage.iterdir()):
            if entry.is_file() and entry.suffix == ".mp4":
                items.append((entry, "video"))
            elif self.dot_ids and self._is_dot_directory(entry):
                items.append((entry, "dot"))
        
        if items:
            n_videos = sum(1 for _, t in items if t == "video")
            n_dots = sum(1 for _, t in items if t == "dot")
            parts = []
            if n_videos:
                parts.append(f"{n_videos} video(s)")
            if n_dots:
                parts.append(f"{n_dots} DOT dir(s)")
            logger.info(f"Found {', '.join(parts)} to process")
        
        return items
    
    def _find_dot_directories(self) -> list:
        """Find unprocessed DOT directories in input_storage."""
        if not self.input_storage.exists() or not self.dot_ids:
            return []
        
        return [d for d in sorted(self.input_storage.iterdir())
                if self._is_dot_directory(d)]
    
    def _parse_dot_dir_name(self, dir_name: str):
        """
        Parse a DOT directory name into (dot_id, date_time).
        
        Directory name format: {dot_id}_{YYYYMMDD}_{HHMMSS}
        Returns (dot_id, "YYYYMMDD_HHMMSS") or (None, None).
        """
        for dot_id in self.dot_ids:
            if dir_name.startswith(f"{dot_id}_"):
                date_time = dir_name[len(dot_id) + 1:]
                return dot_id, date_time
        return None, None
    
    def _compute_output_dir(self, device_id: str, date_time: str) -> Path:
        """Compute the output directory for a device and timestamp."""
        results_dir = Path(self.config["output"]["results_dir"])
        return results_dir / device_id / date_time
    
    # ------------------------------------------------------------------
    # Last-recording marker (persists across restarts)
    # ------------------------------------------------------------------
    
    @property
    def _marker_path(self) -> Path:
        return self.input_storage / ".last_recording"
    
    def _load_last_recording_marker(self) -> None:
        """Read the .last_recording marker on startup."""
        if not self._marker_path.exists():
            return
        
        marker_video = self._marker_path.read_text().strip()
        if not marker_video:
            self._marker_path.unlink(missing_ok=True)
            return
        
        if (self.input_storage / marker_video).exists():
            # Video still waiting to be processed
            self._reset_after_video = marker_video
            logger.info(f"Previous session marker: will reset tracker after {marker_video}")
        else:
            # Already processed (deleted) — reset before next video
            self._pending_tracker_reset = True
            self._marker_path.unlink(missing_ok=True)
            logger.info(f"Previous session ended ({marker_video} already processed), "
                       f"tracker will reset on next video")
    
    def _save_last_recording_marker(self) -> None:
        """Write the .last_recording marker when recording stops."""
        if not (self.continuous_tracking and self.recorder
                and self.recorder.last_chunk_path):
            return
        
        filename = self.recorder.last_chunk_path.name
        self._marker_path.write_text(filename)
        self._reset_after_video = filename
        logger.info(f"Marked last recording: {filename}")
    
    def _clear_last_recording_marker(self) -> None:
        """Delete the marker after the boundary video is processed."""
        self._marker_path.unlink(missing_ok=True)
        self._reset_after_video = ""
    
    # ------------------------------------------------------------------
    
    def _processor_worker(self) -> None:
        """
        Worker that processes video chunks and DOT directories.
        
        Items from input_storage are processed in chronological order
        (sorted by name). DOT classification does NOT touch the BugSpot
        tracker, so continuous tracking across FLICK videos is preserved
        even when a DOT directory is processed in between.
        """
        logger.info("Processor started")
        
        # Process existing items in chronological order
        for path, item_type in self._find_existing_items():
            if self.stop_event.is_set():
                break
            if item_type == "video":
                self._process_video(path)
            else:
                self._process_dot_directory(path)
        
        # Process new videos from queue + poll for DOT directories
        while not self.stop_event.is_set():
            try:
                video_path = self.video_queue.get(timeout=1.0)
                self._process_video(video_path)
                self.video_queue.task_done()
            except queue.Empty:
                # Check for new DOT directories while waiting
                for dot_dir in self._find_dot_directories():
                    if self.stop_event.is_set():
                        break
                    self._process_dot_directory(dot_dir)
                
                # If recording stopped, check if we're done
                if self.recording_stopped.is_set():
                    remaining = self.video_queue.qsize()
                    remaining_dots = len(self._find_dot_directories())
                    if remaining == 0 and remaining_dots == 0:
                        logger.info("Queue empty - processing complete")
                        break
                continue
            except Exception as e:
                logger.error(f"Processor error: {e}", exc_info=True)
        
        logger.info("Processor stopped")
    
    def _process_video(self, video_path: Path) -> None:
        """Process a single video file from this FLICK."""
        if not video_path.exists():
            logger.warning(f"Video not found: {video_path}")
            return
        
        logger.info("-" * 50)
        logger.info(f"PROCESSING: {video_path.name}")
        logger.info("-" * 50)
        
        try:
            # Compute output directory: results_dir/flick_id/date_time/
            date_time = video_path.stem[len(self.flick_id) + 1:]
            output_dir = self._compute_output_dir(self.flick_id, date_time)
            
            # --- Pre-process tracker resets (continuous_tracking only) ---
            if self.continuous_tracking:
                # Pending reset from a previous session whose marker video
                # was already processed before we started
                if self._pending_tracker_reset:
                    logger.info("Resetting tracker (previous recording session ended)")
                    self.processor.reset_tracker()
                    self._pending_tracker_reset = False
                
                # Day-change detection
                video_date = date_time[:8]  # YYYYMMDD
                if self._last_video_date and video_date != self._last_video_date:
                    logger.info(f"Day changed ({self._last_video_date} → {video_date}), resetting tracker")
                    self.processor.reset_tracker()
                self._last_video_date = video_date
            
            # Process
            results = self.processor.process_video(video_path, output_dir)
            
            # Write results JSON
            output_paths = self.writer.write_results(
                results=results,
                output_dir=output_dir,
            )
            
            # Summary
            summary = results.get("summary", {})
            logger.info(f"COMPLETE: {summary.get('confirmed_tracks', 0)} insects confirmed "
                       f"({summary.get('total_tracks', 0)} total tracks)")
            
            if output_paths.get('json'):
                logger.info(f"Output: {output_paths['json']}")
            
            # Cleanup
            self.processor.clear_video_detections()
            self._delete_video(video_path)
            
            # Recording-stop boundary: reset tracker after the last
            # video from the previous recording session
            if self._reset_after_video and video_path.name == self._reset_after_video:
                logger.info(f"Last recorded video processed, resetting tracker")
                self.processor.reset_tracker()
                self._clear_last_recording_marker()
            
        except Exception as e:
            logger.error(f"Failed to process {video_path.name}: {e}", exc_info=True)
    
    def _process_dot_directory(self, dot_dir: Path) -> None:
        """
        Process a DOT device directory.
        
        If classification is enabled: classify crops → write results.json.
        If classification is disabled: copy crops + composites only (no results).
        
        DOT directory structure:
            {dot_id}_{date}_{time}/
                {dot_id}_{date}_{time}_crops/
                    {track_id}/
                        frame_000075.jpg
                        ...
                {dot_id}_{date}_{time}_composites/
                    ...
        """
        logger.info("-" * 50)
        logger.info(f"PROCESSING DOT: {dot_dir.name}")
        logger.info("-" * 50)
        
        try:
            dot_id, date_time = self._parse_dot_dir_name(dot_dir.name)
            if not dot_id:
                logger.warning(f"Could not parse DOT directory: {dot_dir.name}")
                return
            
            # Compute output directory
            output_dir = self._compute_output_dir(dot_id, date_time)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy crops and composites to output
            src_crops = dot_dir / f"{dot_dir.name}_crops"
            src_composites = dot_dir / f"{dot_dir.name}_composites"
            
            if src_crops.exists():
                shutil.copytree(src_crops, output_dir / "crops", dirs_exist_ok=True)
                logger.info(f"  Copied crops to {output_dir / 'crops'}")
            
            if src_composites.exists():
                shutil.copytree(src_composites, output_dir / "composites", dirs_exist_ok=True)
                logger.info(f"  Copied composites to {output_dir / 'composites'}")
            
            # Classify and write results (only if classification enabled)
            if self.enable_classification:
                results = self.processor.classify_dot_directory(dot_dir)
                
                self.writer.write_results(
                    results=results,
                    output_dir=output_dir,
                )
                
                summary = results.get("summary", {})
                logger.info(f"DOT COMPLETE: {summary.get('confirmed_tracks', 0)} tracks classified "
                           f"from {dot_id}")
            else:
                logger.info(f"DOT COMPLETE: copied crops/composites from {dot_id} "
                           f"(classification disabled, no results.json)")
            
            # Delete inbox directory
            shutil.rmtree(dot_dir)
            logger.debug(f"Deleted DOT directory: {dot_dir.name}")
            
        except Exception as e:
            logger.error(f"Failed to process DOT {dot_dir.name}: {e}", exc_info=True)
    
    def _delete_video(self, video_path: Path) -> None:
        """Delete processed video."""
        try:
            video_path.unlink()
            logger.debug(f"Deleted: {video_path.name}")
        except Exception as e:
            logger.error(f"Could not delete {video_path.name}: {e}")
    
    def start(self) -> None:
        """Start the pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING PIPELINE")
        logger.info("=" * 60)
        
        # Start recorder (if enabled)
        if self.enable_recording and self.recorder:
            self.recorder_thread = threading.Thread(
                target=self.recorder.start,
                daemon=True,
                name="Recorder"
            )
            self.recorder_thread.start()
            logger.info("Recorder thread started")
        else:
            self.recording_stopped.set()  # No recording
        
        # Start processor (if enabled)
        if self.enable_processing and self.processor:
            self.processor_thread = threading.Thread(
                target=self._processor_worker,
                daemon=False,  # Non-daemon so it completes
                name="Processor"
            )
            self.processor_thread.start()
            logger.info("Processor thread started")
        
        if self.enable_recording and self.enable_processing:
            logger.info("Pipeline running - Ctrl+C to stop recording (processing continues)")
        elif self.enable_recording:
            logger.info("Recording - Ctrl+C to stop")
        else:
            logger.info("Processing existing videos...")
    
    def stop_recording(self) -> None:
        """Stop recording only, processing continues."""
        if not self.recording_stopped.is_set():
            logger.info("=" * 60)
            logger.info("STOPPING RECORDING")
            logger.info("=" * 60)
            
            if self.recorder:
                self.recorder.stop()
            if self.recorder_thread:
                self.recorder_thread.join(timeout=10.0)
            
            # Mark the last recorded video so tracker resets after it
            self._save_last_recording_marker()
            
            self.recording_stopped.set()
            logger.info("Recording stopped - processing remaining videos...")
            
            remaining = self.video_queue.qsize()
            if remaining > 0:
                logger.info(f"Videos in queue: {remaining}")
    
    def stop(self) -> None:
        """Stop the pipeline gracefully."""
        logger.info("=" * 60)
        logger.info("STOPPING PIPELINE")
        logger.info("=" * 60)
        
        # Stop recorder first
        self.stop_recording()
        
        # Stop processor
        self.stop_event.set()
        if self.processor_thread:
            self.processor_thread.join(timeout=30.0)
            logger.info("Processor stopped")
        
        logger.info("Pipeline stopped cleanly")
    
    def wait(self) -> None:
        """Wait for pipeline (blocks until stopped)."""
        # Wait for recorder to finish (if running)
        if self.recorder_thread:
            self.recorder_thread.join()
        
        # Wait for processor to finish (if running)
        if self.processor_thread:
            self.processor_thread.join()


def load_config(config_path: str) -> dict:
    """Load and validate configuration."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(path) as f:
        config = yaml.safe_load(f)
    
    # Required sections
    required = ["device", "paths", "capture", "detection", "classification", "tracking", "output"]
    missing = [s for s in required if s not in config]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")
    
    # Required device keys
    if "flick_id" not in config["device"]:
        raise ValueError("Missing device.flick_id in config")
    
    # Required capture keys
    capture_keys = ["camera_index", "use_picamera", "fps", "chunk_duration_seconds"]
    for key in capture_keys:
        if key not in config["capture"]:
            raise ValueError(f"Missing capture.{key} in config")
    
    # Required tracking keys
    tracking_keys = ["max_lost_frames", "w_dist", "w_area", "cost_threshold"]
    for key in tracking_keys:
        if key not in config["tracking"]:
            raise ValueError(f"Missing tracking.{key} in config")
    
    # Required detection keys
    detection_keys = ["min_area", "max_area", "min_density", "min_solidity",
                      "min_largest_blob_ratio", "max_num_blobs", "min_displacement",
                      "min_path_points", "max_frame_jump", "max_revisit_ratio",
                      "min_progression_ratio", "max_directional_variance"]
    for key in detection_keys:
        if key not in config["detection"]:
            raise ValueError(f"Missing detection.{key} in config")
    
    # Resolve paths relative to edge26/
    config_dir = path.parent.parent
    
    # Model path
    if "classification" in config and "model" in config["classification"]:
        model_path = config["classification"]["model"]
        if not Path(model_path).is_absolute():
            config["classification"]["model"] = str(config_dir / model_path)
    
    # Output path
    if "output" in config and "results_dir" in config["output"]:
        results_dir = config["output"]["results_dir"]
        if not Path(results_dir).is_absolute():
            config["output"]["results_dir"] = str(config_dir / results_dir)
    
    # Logs path
    if "paths" in config:
        logs_dir = config["paths"].get("logs_dir", "output/logs")
        if not Path(logs_dir).is_absolute():
            config["paths"]["logs_dir"] = str(config_dir / logs_dir)
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Edge26 - On-Device Insect Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --config config/settings.yaml
  python main.py -c config/settings.yaml
        """
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to settings.yaml configuration file"
    )
    args = parser.parse_args()
    
    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    
    # Setup logging
    setup_logging(Path(config["paths"]["logs_dir"]))
    
    # Banner
    logger.info("=" * 60)
    logger.info("  EDGE26 - On-Device Insect Detection Pipeline")
    logger.info("=" * 60)
    
    # Initialize pipeline
    try:
        pipeline = Pipeline(config=config)
    except Exception as e:
        logger.error(f"Failed to initialize: {e}", exc_info=True)
        return 1
    
    # Track shutdown state
    shutdown_count = [0]
    
    def signal_handler(signum, frame):
        shutdown_count[0] += 1
        
        if shutdown_count[0] == 1:
            # First Ctrl+C: Stop recording, continue processing
            logger.info("Ctrl+C received - stopping recording")
            pipeline.stop_recording()
        else:
            # Second Ctrl+C: Force stop everything immediately
            logger.info("Second Ctrl+C - forcing shutdown")
            pipeline.stop_event.set()
            os._exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run
    try:
        pipeline.start()
        pipeline.wait()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        pipeline.stop()
        return 1
    
    logger.info("Pipeline completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Edge26 - On-Device Insect Detection Pipeline

Usage:
    python main.py --config config/settings.yaml
"""

import argparse
import logging
import queue
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
        
        # Pipeline mode
        pipeline_config = config.get("pipeline", {})
        self.enable_recording = pipeline_config.get("enable_recording", True)
        self.enable_processing = pipeline_config.get("enable_processing", True)
        
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
        logger.info(f"Video storage: {config['paths']['video_storage']}")
        if self.enable_processing:
            logger.info(f"Results dir:   {config['output']['results_dir']}")
        if self.enable_recording:
            logger.info(f"Chunk duration: {config['capture']['chunk_duration_seconds']}s")
    
    def _init_recorder(self) -> VideoRecorder:
        """Initialize video recorder from config."""
        paths = self.config["paths"]
        capture = self.config["capture"]
        
        return VideoRecorder(
            output_dir=paths["video_storage"],
            fps=capture["fps"],
            chunk_duration=capture["chunk_duration_seconds"],
            device_id=capture["device_id"],
            video_queue=self.video_queue,
            camera_index=capture["camera_index"],
            use_picamera=capture["use_picamera"],
        )
    
    def _find_existing_videos(self) -> list:
        """Find existing videos in storage."""
        video_storage = Path(self.config["paths"]["video_storage"])
        
        if not video_storage.exists():
            return []
        
        videos = sorted(video_storage.glob("*.mp4"))
        if videos:
            logger.info(f"Found {len(videos)} existing video(s) to process")
        
        return videos
    
    def _processor_worker(self) -> None:
        """Worker that processes video chunks."""
        logger.info("Processor started")
        
        # Process existing videos first
        for video_path in self._find_existing_videos():
            if self.stop_event.is_set():
                break
            self._process_video(video_path)
        
        # Process new videos from queue
        while not self.stop_event.is_set():
            try:
                video_path = self.video_queue.get(timeout=1.0)
                self._process_video(video_path)
                self.video_queue.task_done()
            except queue.Empty:
                # If recording stopped, check if we're done
                if self.recording_stopped.is_set():
                    remaining = self.video_queue.qsize()
                    if remaining == 0:
                        logger.info("Queue empty - processing complete")
                        break
                continue
            except Exception as e:
                logger.error(f"Processor error: {e}", exc_info=True)
        
        logger.info("Processor stopped")
    
    def _process_video(self, video_path: Path) -> None:
        """Process a single video file."""
        if not video_path.exists():
            logger.warning(f"Video not found: {video_path}")
            return
        
        logger.info("-" * 50)
        logger.info(f"PROCESSING: {video_path.name}")
        logger.info("-" * 50)
        
        try:
            # Process
            results = self.processor.process_video(video_path)
            
            # Write results
            output_paths = self.writer.write_results(
                results=results,
                video_path=video_path,
                processor=self.processor
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
            
        except Exception as e:
            logger.error(f"Failed to process {video_path.name}: {e}", exc_info=True)
    
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
    required = ["paths", "capture", "detection", "classification", "tracking", "output"]
    missing = [s for s in required if s not in config]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")
    
    # Required capture keys
    capture_keys = ["device_id", "camera_index", "use_picamera", "fps", "chunk_duration_seconds"]
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
            # Second Ctrl+C: Force stop everything
            logger.info("Second Ctrl+C - forcing shutdown")
            pipeline.stop()
            sys.exit(0)
    
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

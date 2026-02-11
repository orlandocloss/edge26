"""
Hierarchical insect classification.

Classifies insects at 3 levels: Family, Genus, Species.
Uses Hailo HEF models for inference.

Matches exact functionality of inference.py HierarchicalInsectClassifier.
"""

import cv2
import numpy as np
import logging
import requests
import sys
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalClassification:
    """Classification result with family, genus, species predictions."""
    family: str
    genus: str
    species: str
    family_confidence: float
    genus_confidence: float
    species_confidence: float
    family_probs: List[float] = field(default_factory=list)
    genus_probs: List[float] = field(default_factory=list)
    species_probs: List[float] = field(default_factory=list)


def get_taxonomy(species_list):
    """
    Retrieves taxonomic information for a list of species from GBIF API.
    Creates a hierarchical taxonomy dictionary with family, genus, and species relationships.
    """
    taxonomy = {1: [], 2: {}, 3: {}}
    
    species_list_for_gbif = [s for s in species_list if s.lower() != 'unknown']
    has_unknown = len(species_list_for_gbif) != len(species_list)
    
    logger.info(f"Building taxonomy from GBIF for {len(species_list_for_gbif)} species")
    
    print("\nTaxonomy Results:")
    print("-" * 80)
    print(f"{'Species':<30} {'Family':<20} {'Genus':<20} {'Status'}")
    print("-" * 80)
    
    for species_name in species_list_for_gbif:
        url = f"https://api.gbif.org/v1/species/match?name={species_name}&verbose=true"
        try:
            response = requests.get(url)
            data = response.json()
            
            if data.get('status') == 'ACCEPTED' or data.get('status') == 'SYNONYM':
                family = data.get('family')
                genus = data.get('genus')
                
                if family and genus:
                    status = "OK"
                    
                    print(f"{species_name:<30} {family:<20} {genus:<20} {status}")
                    
                    taxonomy[3][species_name] = genus
                    taxonomy[2][genus] = family
                    
                    if family not in taxonomy[1]:
                        taxonomy[1].append(family)
                else:
                    error_msg = f"Species '{species_name}' found in GBIF but family and genus not found, could be spelling error in species, check GBIF"
                    logger.error(error_msg)
                    print(f"{species_name:<30} {'Not found':<20} {'Not found':<20} ERROR")
                    print(f"Error: {error_msg}")
                    sys.exit(1)
            else:
                error_msg = f"Species '{species_name}' not found in GBIF, could be spelling error, check GBIF"
                logger.error(error_msg)
                print(f"{species_name:<30} {'Not found':<20} {'Not found':<20} ERROR")
                print(f"Error: {error_msg}")
                sys.exit(1)
                
        except Exception as e:
            error_msg = f"Error retrieving data for species '{species_name}': {str(e)}"
            logger.error(error_msg)
            print(f"{species_name:<30} {'Error':<20} {'Error':<20} FAILED")
            print(f"Error: {error_msg}")
            sys.exit(1)

    if has_unknown:
        unknown_family = "Unknown"
        unknown_genus = "Unknown"
        unknown_species = "unknown"
        
        if unknown_family not in taxonomy[1]:
            taxonomy[1].append(unknown_family)
        
        taxonomy[2][unknown_genus] = unknown_family
        taxonomy[3][unknown_species] = unknown_genus
        
        print(f"{unknown_species:<30} {unknown_family:<20} {unknown_genus:<20} {'OK'}")
    
    taxonomy[1] = sorted(list(set(taxonomy[1])))
    print("-" * 80)
    
    num_families = len(taxonomy[1])
    num_genera = len(taxonomy[2])
    num_species = len(taxonomy[3])
    
    print("\nFamily indices:")
    for i, family in enumerate(taxonomy[1]):
        print(f"  {i}: {family}")
    
    print("\nGenus indices:")
    for i, genus in enumerate(sorted(taxonomy[2].keys())):
        print(f"  {i}: {genus}")
    
    print("\nSpecies indices:")
    for i, species in enumerate(species_list):
        print(f"  {i}: {species}")
    
    logger.info(f"Taxonomy built: {num_families} families, {num_genera} genera, {num_species} species")
    return taxonomy


class HailoClassifier:
    """
    Hailo-based hierarchical insect classifier.
    
    Loads a HEF model and runs classification inference.
    Outputs predictions for family, genus, and species.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the Hailo classifier.
        
        Args:
            config: Classification configuration from settings.yaml
        """
        self.config = config
        self.model_path = Path(config["model"])
        
        # Input size will be read from model, or use config override
        self._input_size = config.get("input_size")  # None = read from model
        
        # Normalization (ImageNet) - only if model expects normalized input
        self.normalize = config.get("normalize", False)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # Hailo components (lazy loaded)
        self._hef = None
        self._vdevice = None
        self._infer_model = None
        self._configured_model = None
        
        # Class labels (loaded from model metadata or file)
        self.family_list: List[str] = []
        self.genus_list: List[str] = []
        self.species_list: List[str] = []
        
        # Taxonomy mappings for hierarchical aggregation
        self.species_to_genus: Dict[str, str] = {}
        self.genus_to_family: Dict[str, str] = {}
        
        logger.info(f"HailoClassifier initialized with model: {self.model_path}")
    
    def _load_model(self) -> None:
        """Load the Hailo model and extract labels."""
        if self._hef is not None:
            return
        
        from hailo_platform import HEF, VDevice, HailoSchedulingAlgorithm
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load HEF
        self._hef = HEF(str(self.model_path))
        
        # Create virtual device
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self._vdevice = VDevice(params)
        
        # Create inference model
        self._infer_model = self._vdevice.create_infer_model(str(self.model_path))
        self._infer_model.set_batch_size(1)
        
        # Load labels and taxonomy
        self._load_labels()
        
        # Configure model for inference
        self._configured_model = self._infer_model.configure()
        
        logger.info(f"Model loaded: families={len(self.family_list)}, "
                   f"genera={len(self.genus_list)}, species={len(self.species_list)}")
    
    def _load_labels(self) -> None:
        """
        Load species labels from a plain text file (one species per line),
        then build taxonomy (family/genus) from GBIF API.
        """
        labels_path = self.config.get("labels")
        if not labels_path:
            logger.warning("No 'labels' path in classification config - using numeric indices")
            self._load_labels_fallback()
            return
        
        labels_path = Path(labels_path)
        if not labels_path.exists():
            logger.warning(f"Labels file not found: {labels_path} - using numeric indices")
            self._load_labels_fallback()
            return
        
        # Read species list (one per line, same order as training)
        with open(labels_path) as f:
            self.species_list = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(self.species_list)} species from {labels_path}")
        
        # Build taxonomy from GBIF
        taxonomy = get_taxonomy(self.species_list)
        
        self.family_list = taxonomy[1]                          # sorted families
        self.genus_list = sorted(taxonomy[2].keys())            # sorted genera
        self.species_to_genus = taxonomy[3]                     # species -> genus
        self.genus_to_family = taxonomy[2]                      # genus -> family
        
        logger.info(f"Taxonomy: {len(self.family_list)} families, "
                    f"{len(self.genus_list)} genera, "
                    f"{len(self.species_list)} species")
    
    def _load_labels_fallback(self) -> None:
        """Fall back to numeric indices from model output shapes."""
        output_infos = self._hef.get_output_vstream_infos()
        for i, info in enumerate(output_infos):
            num_classes = info.shape[-1]
            if i == 0:
                self.family_list = [f"family_{j}" for j in range(num_classes)]
            elif i == 1:
                self.genus_list = [f"genus_{j}" for j in range(num_classes)]
            else:
                self.species_list = [f"class_{j}" for j in range(num_classes)]
        
        if len(output_infos) == 1:
            num_classes = output_infos[0].shape[-1]
            self.species_list = [f"class_{j}" for j in range(num_classes)]
            self.family_list = self.species_list
            self.genus_list = self.species_list
    
    def classify(self, crop: np.ndarray) -> HierarchicalClassification:
        """
        Classify a single detection crop.
        
        Args:
            crop: BGR image crop as numpy array
            
        Returns:
            HierarchicalClassification result
        """
        self._load_model()
        
        # Preprocess (match inference.py transforms)
        preprocessed = self._preprocess(crop)
        
        # Run inference
        outputs = self._run_inference(preprocessed)
        
        # Parse 3 output heads (family, genus, species)
        if len(outputs) >= 3:
            family_logits = outputs[0].flatten()
            genus_logits = outputs[1].flatten()
            species_logits = outputs[2].flatten()
        else:
            # Single output - assume species only
            species_logits = outputs[0].flatten()
            family_logits = np.zeros(len(self.family_list)) if self.family_list else species_logits
            genus_logits = np.zeros(len(self.genus_list)) if self.genus_list else species_logits
        
        # Softmax
        family_probs = self._softmax(family_logits)
        genus_probs = self._softmax(genus_logits)
        species_probs = self._softmax(species_logits)
        
        # Get best predictions
        family_idx = int(np.argmax(family_probs))
        genus_idx = int(np.argmax(genus_probs))
        species_idx = int(np.argmax(species_probs))
        
        return HierarchicalClassification(
            family=self.family_list[family_idx] if family_idx < len(self.family_list) else f"Family_{family_idx}",
            genus=self.genus_list[genus_idx] if genus_idx < len(self.genus_list) else f"Genus_{genus_idx}",
            species=self.species_list[species_idx] if species_idx < len(self.species_list) else f"Species_{species_idx}",
            family_confidence=float(family_probs[family_idx]),
            genus_confidence=float(genus_probs[genus_idx]),
            species_confidence=float(species_probs[species_idx]),
            family_probs=family_probs.tolist(),
            genus_probs=genus_probs.tolist(),
            species_probs=species_probs.tolist()
        )
    
    def classify_batch(self, crops: List[np.ndarray]) -> List[HierarchicalClassification]:
        """Classify a batch of detection crops."""
        return [self.classify(crop) for crop in crops]
    
    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess crop for model input.
        
        Simply resizes to model's expected input size.
        No fancy transforms - model was trained on detection crop sizes.
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Get target size from model - shape is (H, W, C) from Hailo vstream info
        input_size = self._get_input_size()
        target_h, target_w = input_size[0], input_size[1]
        
        # Resize directly to model input size
        resized = cv2.resize(rgb, (target_w, target_h))
        
        # Convert to float
        if self.normalize:
            # ImageNet normalization
            normalized = resized.astype(np.float32) / 255.0
            normalized = (normalized - self.mean) / self.std
            return normalized.transpose(2, 0, 1).astype(np.float32)
        else:
            # Hailo models typically expect uint8
            return resized.astype(np.uint8)
    
    def _get_input_size(self) -> Tuple[int, int, int]:
        """Get model input size (H, W, C)."""
        if self._input_size:
            # Config override
            if isinstance(self._input_size, (list, tuple)) and len(self._input_size) >= 2:
                h, w = self._input_size[0], self._input_size[1]
                return (h, w, 3)
        
        # Read from model
        self._load_model()
        input_info = self._hef.get_input_vstream_infos()[0]
        return input_info.shape  # Typically (H, W, C) for Hailo
    
    def _run_inference(self, preprocessed: np.ndarray) -> List[np.ndarray]:
        """Run inference and return outputs."""
        configured = self._configured_model
        
        # Get output info
        output_infos = self._hef.get_output_vstream_infos()
        
    # Create output buffers (zeros so any unwritten region is 0, not NaN)
        output_buffers = {}
        for info in output_infos:
            shape = self._infer_model.output(info.name).shape
        output_buffers[info.name] = np.zeros(shape, dtype=np.float32)
        
        # Create bindings
        bindings = configured.create_bindings(output_buffers=output_buffers)
        
        # Prepare input - ensure correct shape (add batch dim if needed)
        if preprocessed.ndim == 3:
            # HWC -> add batch -> NHWC
            preprocessed = np.expand_dims(preprocessed, 0)
        
        bindings.input().set_buffer(np.ascontiguousarray(preprocessed))
            
        # Run (timeout in milliseconds)
        configured.run([bindings], timeout=10000)
            
        # Read results from the output buffers (run() writes into them in-place)
        outputs = [output_buffers[info.name] for info in output_infos]
        
        return outputs
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Apply softmax to logits."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def hierarchical_aggregate(self, classifications: List[HierarchicalClassification]) -> Dict:
        """
        Aggregate per-frame classifications using hierarchical selection.
        
        Matches inference.py hierarchical_aggregation:
        1. Average probabilities across frames
        2. Select best family
        3. Select best genus WITHIN that family
        4. Select best species WITHIN that genus
        
        Args:
            classifications: List of per-frame classifications
            
        Returns:
            Dict with final hierarchical prediction
        """
        if not classifications:
            return None
        
        # Average probabilities
        family_probs_avg = np.mean([c.family_probs for c in classifications], axis=0)
        genus_probs_avg = np.mean([c.genus_probs for c in classifications], axis=0)
        species_probs_avg = np.mean([c.species_probs for c in classifications], axis=0)
        
        # Best family
        best_family_idx = int(np.argmax(family_probs_avg))
        best_family = self.family_list[best_family_idx] if best_family_idx < len(self.family_list) else f"Family_{best_family_idx}"
        
        # Best genus WITHIN that family
        family_genera = [i for i, g in enumerate(self.genus_list) 
                        if self.genus_to_family.get(g) == best_family]
        if family_genera:
            best_genus_idx = family_genera[int(np.argmax(genus_probs_avg[family_genera]))]
        else:
            best_genus_idx = int(np.argmax(genus_probs_avg))
        best_genus = self.genus_list[best_genus_idx] if best_genus_idx < len(self.genus_list) else f"Genus_{best_genus_idx}"
        
        # Best species WITHIN that genus
        genus_species = [i for i, s in enumerate(self.species_list)
                        if self.species_to_genus.get(s) == best_genus]
        if genus_species:
            best_species_idx = genus_species[int(np.argmax(species_probs_avg[genus_species]))]
        else:
            best_species_idx = int(np.argmax(species_probs_avg))
        best_species = self.species_list[best_species_idx] if best_species_idx < len(self.species_list) else f"Species_{best_species_idx}"
        
        return {
            'family': best_family,
            'genus': best_genus,
            'species': best_species,
            'family_confidence': float(family_probs_avg[best_family_idx]),
            'genus_confidence': float(genus_probs_avg[best_genus_idx]),
            'species_confidence': float(species_probs_avg[best_species_idx]),
        }
    
    @property
    def num_families(self) -> int:
        self._load_model()
        return len(self.family_list)
    
    @property
    def num_genera(self) -> int:
        self._load_model()
        return len(self.genus_list)
    
    @property
    def num_species(self) -> int:
        self._load_model()
        return len(self.species_list)

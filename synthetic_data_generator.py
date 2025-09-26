import numpy as np
import nibabel as nib
import os
from pathlib import Path
import json
from datetime import datetime
from scipy import ndimage
import matplotlib.pyplot as plt

class SyntheticStrokeDataGenerator:
    """Generate synthetic multimodal stroke imaging data"""
    
    def __init__(self, output_dir="data/synthetic"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Standard brain dimensions
        self.brain_shape = (182, 218, 182)  # MNI152 standard
        self.voxel_size = (1.0, 1.0, 1.0)  # mm
        
        # Tissue parameters
        self.tissue_values = {
            'csf': {'t1': 300, 't2': 2000, 'dwi': 200, 'adc': 3.0},
            'gray_matter': {'t1': 800, 't2': 100, 'dwi': 800, 'adc': 0.8},
            'white_matter': {'t1': 600, 't2': 80, 'dwi': 600, 'adc': 0.7},
            'lesion_acute': {'t1': 700, 't2': 150, 'dwi': 1200, 'adc': 0.4},
            'lesion_chronic': {'t1': 300, 't2': 1800, 'dwi': 400, 'adc': 2.0},
            'blood': {'t1': 300, 't2': 30, 'dwi': 100, 'adc': 0.5}
        }
    
    def generate_brain_mask(self):
        """Generate realistic brain mask"""
        # Create elliptical brain shape
        z, y, x = np.ogrid[:self.brain_shape[0], :self.brain_shape[1], :self.brain_shape[2]]
        center_z, center_y, center_x = np.array(self.brain_shape) // 2
        
        # Brain ellipsoid parameters
        a, b, c = 80, 100, 70  # Semi-axes lengths
        
        brain_mask = ((z - center_z)**2 / a**2 + 
                     (y - center_y)**2 / b**2 + 
                     (x - center_x)**2 / c**2) <= 1
        
        # Add some irregularity
        brain_mask = ndimage.binary_opening(brain_mask)
        brain_mask = ndimage.binary_closing(brain_mask)
        
        return brain_mask.astype(np.uint8)
    
    def generate_tissue_segmentation(self, brain_mask):
        """Generate tissue segmentation (CSF, GM, WM)"""
        # Initialize segmentation
        segmentation = np.zeros(self.brain_shape, dtype=np.uint8)
        
        # Create tissue probability maps
        z, y, x = np.ogrid[:self.brain_shape[0], :self.brain_shape[1], :self.brain_shape[2]]
        center_z, center_y, center_x = np.array(self.brain_shape) // 2
        
        # Distance from center
        dist_from_center = np.sqrt((z - center_z)**2 + (y - center_y)**2 + (x - center_x)**2)
        
        # CSF (ventricles and periphery)
        csf_prob = np.exp(-((dist_from_center - 15)**2) / 200) + \
                  np.exp(-((dist_from_center - 80)**2) / 100)
        
        # Gray matter (cortical)
        gm_prob = np.exp(-((dist_from_center - 60)**2) / 300)
        
        # White matter (central)
        wm_prob = np.exp(-((dist_from_center - 40)**2) / 400)
        
        # Assign tissues based on probabilities
        tissue_probs = np.stack([csf_prob, gm_prob, wm_prob], axis=-1)
        tissue_assignment = np.argmax(tissue_probs, axis=-1)
        
        # Apply brain mask
        segmentation[brain_mask > 0] = tissue_assignment[brain_mask > 0] + 1
        
        return segmentation
    
    def generate_stroke_lesion(self, brain_mask, lesion_type='acute', size='medium'):
        """Generate stroke lesion mask"""
        # Lesion size parameters
        size_params = {
            'small': (5, 15),
            'medium': (15, 40), 
            'large': (40, 80)
        }
        
        min_vol, max_vol = size_params[size]
        lesion_volume = np.random.uniform(min_vol, max_vol)
        
        # Common stroke locations (MCA territory)
        stroke_centers = [
            (90, 140, 110),  # Left MCA
            (90, 80, 110),   # Right MCA
            (110, 110, 90),  # Posterior circulation
        ]
        
        center = stroke_centers[np.random.randint(len(stroke_centers))]
        
        # Generate lesion shape
        lesion_mask = np.zeros(self.brain_shape, dtype=np.uint8)
        
        # Create elliptical lesion
        z, y, x = np.ogrid[:self.brain_shape[0], :self.brain_shape[1], :self.brain_shape[2]]
        
        # Lesion dimensions
        a = np.random.uniform(8, 20)
        b = np.random.uniform(8, 25)  
        c = np.random.uniform(6, 18)
        
        lesion_ellipsoid = ((z - center[0])**2 / a**2 + 
                           (y - center[1])**2 / b**2 + 
                           (x - center[2])**2 / c**2) <= 1
        
        # Add irregular boundaries
        lesion_ellipsoid = ndimage.binary_erosion(lesion_ellipsoid)
        lesion_ellipsoid = ndimage.binary_dilation(lesion_ellipsoid, iterations=2)
        
        # Ensure lesion is within brain
        lesion_mask = lesion_ellipsoid & brain_mask.astype(bool)
        
        return lesion_mask.astype(np.uint8)
    
    def generate_modality_image(self, segmentation, lesion_mask, modality='t1'):
        """Generate specific imaging modality"""
        image = np.zeros(self.brain_shape, dtype=np.float32)
        
        # Base tissue values
        for tissue_id, tissue_name in enumerate(['background', 'csf', 'gray_matter', 'white_matter']):
            if tissue_name in self.tissue_values:
                mask = segmentation == tissue_id
                base_value = self.tissue_values[tissue_name][modality]
                noise = np.random.normal(0, base_value * 0.1, np.sum(mask))
                image[mask] = base_value + noise
        
        # Add lesion
        if np.sum(lesion_mask) > 0:
            lesion_value = self.tissue_values['lesion_acute'][modality]
            noise = np.random.normal(0, lesion_value * 0.15, np.sum(lesion_mask))
            image[lesion_mask > 0] = lesion_value + noise
        
        # Add noise and artifacts
        image = self.add_imaging_artifacts(image, modality)
        
        return image
    
    def add_imaging_artifacts(self, image, modality):
        """Add realistic imaging artifacts"""
        # Gaussian noise
        noise_std = np.mean(image[image > 0]) * 0.05
        image += np.random.normal(0, noise_std, image.shape)
        
        # Bias field (intensity inhomogeneity)
        z, y, x = np.ogrid[:self.brain_shape[0], :self.brain_shape[1], :self.brain_shape[2]]
        bias_field = 1 + 0.2 * np.sin(z / 30) * np.cos(y / 40) * np.sin(x / 35)
        image *= bias_field
        
        # Motion artifacts (occasionally)
        if np.random.random() < 0.1:
            # Simulate motion by slight shifting
            shift = np.random.randint(-2, 3, 3)
            image = ndimage.shift(image, shift, order=1, cval=0)
        
        return image
    
    def generate_perfusion_maps(self, brain_mask, lesion_mask):
        """Generate perfusion maps (CBF, CBV, MTT)"""
        # Normal perfusion values
        normal_cbf = np.random.normal(50, 15, self.brain_shape)  # mL/100g/min
        normal_cbv = np.random.normal(4, 1, self.brain_shape)    # mL/100g
        normal_mtt = np.random.normal(4, 1, self.brain_shape)    # seconds
        
        # Apply brain mask
        normal_cbf[brain_mask == 0] = 0
        normal_cbv[brain_mask == 0] = 0
        normal_mtt[brain_mask == 0] = 0
        
        # Create perfusion deficit in lesion area and penumbra
        if np.sum(lesion_mask) > 0:
            # Core (severe hypoperfusion)
            normal_cbf[lesion_mask > 0] *= 0.1  # 90% reduction
            normal_cbv[lesion_mask > 0] *= 0.3
            normal_mtt[lesion_mask > 0] *= 3.0
            
            # Penumbra (moderate hypoperfusion)
            penumbra_mask = ndimage.binary_dilation(lesion_mask, iterations=3) & ~lesion_mask
            normal_cbf[penumbra_mask] *= 0.4  # 60% reduction
            normal_cbv[penumbra_mask] *= 0.6
            normal_mtt[penumbra_mask] *= 2.0
        
        return {
            'cbf': normal_cbf,
            'cbv': normal_cbv,
            'mtt': normal_mtt
        }
    
    def generate_vessel_map(self, brain_mask):
        """Generate vessel/angiography map"""
        vessel_map = np.zeros(self.brain_shape, dtype=np.float32)
        
        # Major vessels (simplified)
        # Middle cerebral arteries
        vessel_map[85:95, 130:150, 100:120] = 200  # Left MCA
        vessel_map[85:95, 70:90, 100:120] = 200    # Right MCA
        
        # Add branching pattern
        for i in range(10):
            start_y = np.random.randint(80, 140)
            start_x = np.random.randint(90, 130)
            length = np.random.randint(20, 40)
            
            for j in range(length):
                z_pos = 90 + j // 4
                y_pos = start_y + np.random.randint(-2, 3)
                x_pos = start_x + j + np.random.randint(-1, 2)
                
                if (0 <= z_pos < self.brain_shape[0] and 
                    0 <= y_pos < self.brain_shape[1] and 
                    0 <= x_pos < self.brain_shape[2]):
                    vessel_map[z_pos, y_pos, x_pos] = 150 - j * 2
        
        # Add noise
        vessel_map += np.random.normal(0, 10, self.brain_shape)
        vessel_map[brain_mask == 0] = 0
        
        return vessel_map
    
    def save_nifti(self, data, filename, modality='t1'):
        """Save data as NIfTI file"""
        # Create NIfTI image
        affine = np.array([
            [self.voxel_size[0], 0, 0, -self.brain_shape[0]//2 * self.voxel_size[0]],
            [0, self.voxel_size[1], 0, -self.brain_shape[1]//2 * self.voxel_size[1]],
            [0, 0, self.voxel_size[2], -self.brain_shape[2]//2 * self.voxel_size[2]],
            [0, 0, 0, 1]
        ])
        
        nifti_img = nib.Nifti1Image(data, affine)
        nifti_img.to_filename(filename)
    
    def generate_clinical_metadata(self, has_stroke, lesion_mask):
        """Generate realistic clinical metadata"""
        
        # Patient demographics
        age = np.random.randint(45, 85)
        sex = np.random.choice(['M', 'F'])
        
        if has_stroke:
            # Stroke-related clinical scores
            nihss = np.random.randint(1, 25)  # Higher scores for larger lesions
            mrs_baseline = np.random.randint(0, 2)
            
            # Time metrics
            onset_to_imaging = np.random.uniform(0.5, 12.0)  # hours
            
            # Calculate lesion volume
            lesion_volume = np.sum(lesion_mask) * np.prod(self.voxel_size) / 1000  # mL
            
            # Simulate treatment decision
            eligible_factors = []
            if lesion_volume < 70:
                eligible_factors.append("Core volume < 70 mL")
            if onset_to_imaging < 6:
                eligible_factors.append("Within time window")
            if nihss >= 6:
                eligible_factors.append("Significant clinical deficit")
            
            therapy_eligible = len(eligible_factors) >= 2
            
        else:
            # No stroke
            nihss = 0
            mrs_baseline = 0
            onset_to_imaging = 0
            lesion_volume = 0
            therapy_eligible = False
            eligible_factors = []
        
        metadata = {
            'case_info': {
                'has_stroke': has_stroke,
                'generated_date': datetime.now().isoformat()
            },
            'demographics': {
                'age': age,
                'sex': sex
            },
            'clinical_scores': {
                'nihss_admission': nihss,
                'mrs_baseline': mrs_baseline,
                'aspects_score': np.random.randint(7, 11) if has_stroke else 10
            },
            'timing': {
                'onset_to_imaging_hours': onset_to_imaging
            },
            'imaging_findings': {
                'lesion_volume_ml': lesion_volume,
                'hemorrhage_present': False,
                'large_vessel_occlusion': has_stroke and lesion_volume > 15
            },
            'treatment_decision': {
                'therapy_eligible': therapy_eligible,
                'decision_factors': eligible_factors
            },
            'synthetic_parameters': {
                'brain_shape': self.brain_shape,
                'voxel_size': self.voxel_size
            }
        }
        
        return metadata
    
    def generate_complete_dataset(self, num_cases=50):
        """Generate complete synthetic dataset"""
        
        print(f"Generating {num_cases} synthetic stroke cases...")
        
        for case_id in range(num_cases):
            print(f"Generating case {case_id + 1}/{num_cases}")
            
            # Create case directory
            case_dir = self.output_dir / f"case_{case_id:03d}"
            case_dir.mkdir(exist_ok=True)
            
            # Generate brain anatomy
            brain_mask = self.generate_brain_mask()
            tissue_seg = self.generate_tissue_segmentation(brain_mask)
            
            # Generate stroke lesion (80% of cases have stroke)
            has_stroke = np.random.random() < 0.8
            if has_stroke:
                lesion_size = np.random.choice(['small', 'medium', 'large'], 
                                             p=[0.3, 0.5, 0.2])
                lesion_mask = self.generate_stroke_lesion(brain_mask, size=lesion_size)
            else:
                lesion_mask = np.zeros(self.brain_shape, dtype=np.uint8)
            
            # Generate multimodal images
            modalities = {
                't1': self.generate_modality_image(tissue_seg, lesion_mask, 't1'),
                't2': self.generate_modality_image(tissue_seg, lesion_mask, 't2'),
                'dwi': self.generate_modality_image(tissue_seg, lesion_mask, 'dwi'),
                'adc': self.generate_modality_image(tissue_seg, lesion_mask, 'adc'),
                'cta': self.generate_vessel_map(brain_mask)
            }
            
            # Generate perfusion maps
            perfusion_maps = self.generate_perfusion_maps(brain_mask, lesion_mask)
            modalities.update(perfusion_maps)
            
            # Save all modalities
            for modality_name, data in modalities.items():
                filename = case_dir / f"{modality_name}.nii.gz"
                self.save_nifti(data, filename, modality_name)
            
            # Save masks
            self.save_nifti(brain_mask, case_dir / "brain_mask.nii.gz")
            self.save_nifti(tissue_seg, case_dir / "tissue_seg.nii.gz")
            self.save_nifti(lesion_mask, case_dir / "lesion_mask.nii.gz")
            
            # Generate clinical metadata
            metadata = self.generate_clinical_metadata(has_stroke, lesion_mask)
            
            with open(case_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"✅ Generated {num_cases} cases in {self.output_dir}")

# Usage example
if __name__ == "__main__":
    generator = SyntheticStrokeDataGenerator("data/synthetic")
    generator.generate_complete_dataset(num_cases=10)
    print("Synthetic dataset generation complete!")

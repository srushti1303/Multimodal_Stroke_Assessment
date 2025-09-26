import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

class MultimodalStrokeNet(nn.Module):
    """Unified multimodal network for stroke assessment"""
    
    def __init__(self, num_modalities=4, num_classes=7):
        super().__init__()
        
        # Shared encoder for each modality
        self.modality_encoders = nn.ModuleList([
            self._build_3d_encoder() for _ in range(num_modalities)
        ])
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=512, num_heads=8, batch_first=True
        )
        
        # Task-specific heads
        self.therapy_head = nn.Sequential(
            nn.Linear(512 * num_modalities, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.tissue_head = nn.Sequential(
            nn.Linear(512 * num_modalities, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 64 * 32 * 3),  # Segmentation output
        )
        
        self.vessel_head = nn.Sequential(
            nn.Linear(512 * num_modalities, 256),
            nn.ReLU(),
            nn.Linear(256, 5),  # Collateral scores 0-4
            nn.Softmax(dim=1)
        )
        
        self.outcome_head = nn.Sequential(
            nn.Linear(512 * num_modalities, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )
    
    def _build_3d_encoder(self):
        """Build 3D CNN encoder for single modality"""
        return nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(128, 512)
        )
    
    def forward(self, modalities):
        """Forward pass through multimodal network"""
        # Encode each modality
        encoded_features = []
        for i, modality in enumerate(modalities):
            features = self.modality_encoders[i](modality)
            encoded_features.append(features)
        
        # Stack for attention
        features_stack = torch.stack(encoded_features, dim=1)  # [batch, modalities, features]
        
        # Apply cross-modal attention
        attended_features, _ = self.cross_attention(
            features_stack, features_stack, features_stack
        )
        
        # Flatten for task heads
        combined_features = attended_features.flatten(1)
        
        # Task-specific predictions
        outputs = {
            "therapy_eligibility": self.therapy_head(combined_features),
            "tissue_segmentation": self.tissue_head(combined_features).view(
                -1, 3, 64, 64, 32
            ),
            "vessel_assessment": self.vessel_head(combined_features),
            "outcome_prediction": self.outcome_head(combined_features)
        }
        
        return outputs

class StrokeModelManager:
    """Manager class for loading and running stroke assessment models"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        """Load pre-trained model"""
        self.model = MultimodalStrokeNet()
        
        # In practice, load actual weights:
        # checkpoint = torch.load(self.config.MODEL_CONFIGS["model_path"])
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, modalities: Dict[str, np.ndarray]) -> Dict:
        """Run inference on multimodal data"""
        if self.model is None:
            self.load_model()
        
        # Preprocess modalities
        processed_modalities = []
        for modality_name in ["NCCT", "DWI", "CTA", "Perfusion"]:
            if modality_name in modalities:
                data = modalities[modality_name]
                # Normalize and convert to tensor
                data = (data - data.mean()) / (data.std() + 1e-8)
                tensor = torch.FloatTensor(data).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, D]
                processed_modalities.append(tensor.to(self.device))
            else:
                # Handle missing modality with zeros
                tensor = torch.zeros(1, 1, 64, 64, 32).to(self.device)
                processed_modalities.append(tensor)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(processed_modalities)
        
        # Post-process outputs
        results = {
            "therapy_eligibility": {
                "eligible": outputs["therapy_eligibility"].item() > 0.5,
                "confidence": outputs["therapy_eligibility"].item(),
                "factors": []  # Will be filled based on other outputs
            },
            "tissue_analysis": self._extract_tissue_volumes(outputs["tissue_segmentation"]),
            "vascular_assessment": self._extract_vessel_info(outputs["vessel_assessment"]),
            "prognosis": self._extract_prognosis(outputs["outcome_prediction"])
        }
        
        return results
    
    def _extract_tissue_volumes(self, segmentation_output):
        """Extract tissue volumes from segmentation"""
        seg = segmentation_output.cpu().numpy()[0]  # Remove batch dim
        seg_classes = np.argmax(seg, axis=0)
        
        voxel_volume = 1.0  # mm³ per voxel
        core_volume = np.sum(seg_classes == 1) * voxel_volume / 1000  # Convert to mL
        penumbra_volume = np.sum(seg_classes == 2) * voxel_volume / 1000
        
        mismatch_ratio = penumbra_volume / max(core_volume, 1)
        mismatch_eligible = (mismatch_ratio > 1.8 and core_volume < 70)
        
        return {
            "core_volume": core_volume,
            "penumbra_volume": penumbra_volume,
            "mismatch_ratio": mismatch_ratio,
            "mismatch_eligible": mismatch_eligible
        }
    
    def _extract_vessel_info(self, vessel_output):
        """Extract vascular information"""
        probs = vessel_output.cpu().numpy()[0]
        collateral_score = np.argmax(probs)
        
        # Simulate additional vessel info
        occlusion_locations = ["M1 MCA", "M2 MCA", "ICA terminus", "Basilar", "None detected"]
        occlusion_location = np.random.choice(occlusion_locations)
        
        recanalization_prob = 0.3 + (collateral_score / 4.0) * 0.6
        
        return {
            "collateral_score": collateral_score,
            "occlusion_location": occlusion_location,
            "recanalization_probability": recanalization_prob
        }
    
    def _extract_prognosis(self, outcome_output):
        """Extract prognostic information"""
        probs = outcome_output.cpu().numpy()[0]
        predicted_mrs = np.argmax(probs)
        
        # Calculate risks based on mRS prediction
        mortality_risk = 0.05 + (predicted_mrs / 6.0) * 0.35
        functional_independence = max(0, 1 - (predicted_mrs / 3.0))
        
        return {
            "mrs_90d": predicted_mrs,
            "mortality_risk": mortality_risk,
            "functional_independence": functional_independence,
            "mrs_probabilities": probs.tolist()
        }

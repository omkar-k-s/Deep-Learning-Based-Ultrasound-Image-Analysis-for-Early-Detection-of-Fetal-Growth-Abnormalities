"""
CNN Model Module for Fetal Ultrasound Classification
======================================================
This module handles:
1. Loading pretrained ResNet/MobileNet model
2. Fine-tuning for 3-class classification:
   - Normal Fetus (Class 0)
   - Fetal Growth Restriction/FGR (Class 1)
   - Other Fetal Abnormalities (Class 2)
3. Feature extraction for RAG module
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np


class FetalUltrasoundCNN:
    """
    Wrapper class for fetal ultrasound classification using pretrained CNN.
    Uses ResNet50 by default (can be swapped with MobileNet for lighter inference).
    """
    
    def __init__(self, model_name='resnet50', num_classes=3, device='cpu'):
        """
        Initialize the CNN model.
        
        Args:
            model_name (str): 'resnet50' or 'mobilenet_v2'
            num_classes (int): Number of output classes (3 for normal/FGR/abnormal)
            device (str): 'cpu' or 'cuda'
        """
        self.device = device
        self.num_classes = num_classes
        self.model_name = model_name
        self.checkpoint_loaded = False
        self.checkpoint_epoch = None
        self.checkpoint_val_acc = None
        
        # Class labels for interpretation
        self.class_labels = {
            0: "Normal Fetus",
            1: "Fetal Growth Restriction (FGR)",
            2: "Other Fetal Abnormalities"
        }
        
        self.condition_descriptions = {
            0: (
                "The fetus appears to be developing normally with appropriate growth parameters. "
                "All key measurements — head circumference, abdominal circumference, and femur length — "
                "are within the healthy range for this stage of pregnancy. The placenta is functioning "
                "well, delivering adequate nutrients and oxygen to support the baby's growth."
            ),
            1: (
                "Signs of restricted fetal growth (FGR) detected. The baby's abdominal circumference "
                "is smaller than expected for this stage of pregnancy, suggesting the baby is not receiving "
                "enough nutrients through the placenta. The baby's head circumference may be relatively "
                "preserved (brain-sparing effect) — a natural protective response where blood flow is "
                "prioritised to the brain over the body. This is a warning sign of placental insufficiency "
                "and requires close monitoring."
            ),
            2: (
                "Unusual anatomical features were detected in this ultrasound image. One or more "
                "measurements or structural features appear outside the normal range in a pattern that "
                "does not match typical growth restriction. This may indicate a structural abnormality, "
                "chromosomal difference, or other condition. Immediate specialist evaluation is required "
                "to determine the exact nature and clinical significance of these findings."
            )
        }
        
        # Load model & weights
        self.model = self._load_pretrained_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_pretrained_model(self):
        """Load pretrained model with custom classifier head and restore checkpoint if available."""
        if self.model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            # Replace the final fully connected layer with Dropout + Linear matching train.py
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=0.4),
                nn.Linear(in_features, self.num_classes)
            )
        elif self.model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, self.num_classes)
        else:
            raise ValueError(f"Model {self.model_name} not supported")
        
        # Automatically check and load trained checkpoint if present
        ckpt_path = Path(__file__).parent / "checkpoint" / "best_model.pth"
        if ckpt_path.exists():
            try:
                print(f"[CNN] Loading trained weights from {ckpt_path}...")
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                state_dict = ckpt.get("state_dict", ckpt)
                model.load_state_dict(state_dict)
                self.checkpoint_loaded = True
                self.checkpoint_epoch = ckpt.get("epoch", None)
                self.checkpoint_val_acc = ckpt.get("val_accuracy", None)
                val_str = f", Val Acc: {self.checkpoint_val_acc:.2f}%" if self.checkpoint_val_acc is not None else ""
                print(f"[CNN] Successfully loaded checkpoint (Epoch: {self.checkpoint_epoch}{val_str})")
            except Exception as e:
                print(f"[CNN] Warning: Could not load checkpoint weights: {e}")
                self.checkpoint_loaded = False
        else:
            print("[CNN] No custom checkpoint found, running with ImageNet base weights.")
            self.checkpoint_loaded = False

        return model
    
    def predict(self, image_path):
        """
        Predict fetal condition from ultrasound image.
        
        Args:
            image_path (str): Path to ultrasound image
            
        Returns:
            dict: Contains prediction, confidence, class label, and condition description
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get prediction
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, pred_class].item() * 100
            
            # Return results
            return {
                'predicted_class': pred_class,
                'class_label': self.class_labels[pred_class],
                'confidence': confidence,
                'all_probabilities': {
                    self.class_labels[i]: probabilities[0, i].item() * 100
                    for i in range(self.num_classes)
                },
                'condition_description': self.condition_descriptions[pred_class],
                'embedding': self._extract_embedding(image_tensor),
                'success': True
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_embedding(self, image_tensor):
        """
        Extract feature embedding from penultimate layer.
        Used for RAG retrieval.
        
        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor
            
        Returns:
            np.ndarray: Feature embedding vector
        """
        # Register hook to capture features from penultimate layer
        features = []
        
        def hook_fn(module, input, output):
            features.append(output.detach().cpu().numpy())
        
        if self.model_name == 'resnet50':
            hook_handle = self.model.avgpool.register_forward_hook(hook_fn)
        else:
            hook_handle = self.model.features.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            _ = self.model(image_tensor)
        
        hook_handle.remove()
        
        # Flatten and return embedding
        embedding = features[0].flatten()
        return embedding
    
    def get_model_info(self):
        """Return information about the model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': self.device,
            'checkpoint_loaded': self.checkpoint_loaded,
            'checkpoint_epoch': self.checkpoint_epoch,
            'checkpoint_val_acc': round(self.checkpoint_val_acc, 2) if self.checkpoint_val_acc is not None else None
        }

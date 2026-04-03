"""
EEG-based Regression with Conformer Architecture (Unified Implementation)

This module implements a Conformer model for predicting continuous values from EEG data.
The architecture combines CNN feature extraction with Transformer sequence modeling.
This is a unified implementation combining the best features from multiple versions.
"""

# Standard library imports
import argparse
import os
import glob
import random
import datetime
import time
import sys
import re
from typing import Tuple, List, Optional, Union, Dict

# Scientific and data processing
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
from torch.backends import cudnn
import torch.nn.init as init

# Einops for tensor manipulation
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

# Set up CUDA devices
gpus = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))


class Config:
    """Configuration class to centralize hyperparameters and settings."""
    
    # Model parameters
    EMB_SIZE = 40             # Embedding size for the model
    DEPTH = 2                 # Number of transformer blocks
    NUM_HEADS = 2             # Number of attention heads
    DROPOUT = 0.5             # Dropout rate
    FORWARD_EXPANSION = 4     # Expansion factor for feedforward network
    
    # Training parameters
    BATCH_SIZE = 16           # Batch size for training
    NUM_EPOCHS = 3000         # Number of training epochs
    LEARNING_RATE = 0.0002    # Learning rate
    BETA1 = 0.5               # Beta1 for Adam optimizer
    BETA2 = 0.999             # Beta2 for Adam optimizer
    
    # Default data parameters - can be overridden via command line
    DEFAULT_EEG_DATA_ROOT = "data/eeg_clustered"
    DEFAULT_LABEL_DATA_ROOT = "data/eeg_original"
    
    # EEG data shape parameters
    EEG_CHANNELS = 29         # Number of EEG channels
    EEG_TIME_POINTS = 80000   # Number of time points in EEG data
    
    # CNN architecture parameters
    CNN_CHANNELS = 40         # Number of CNN channels
    TEMP_KERNEL_SIZE = (1, 25)     # Temporal convolution kernel size
    TEMP_STRIDE = (1, 15)          # Temporal convolution stride
    SPATIAL_KERNEL_SIZE = (EEG_CHANNELS, 1)  # Spatial convolution kernel size
    SPATIAL_STRIDE = (1, 1)        # Spatial convolution stride
    POOL_KERNEL_SIZE = (1, 85)     # Pooling kernel size
    POOL_STRIDE = (1, 85)          # Pooling stride
    
    # Prediction projection parameters (for discrete ranges)
    # The values below define the range floors and uppers for projection
    RANGE_FLOORS = [0.0, 0.00, 2.99, 9.64, 16.00, 21.33, 27.97, 36.46, 47.76, 65.06, 88.64, 130.46]
    RANGE_UPPERS = [0.0, 2.99, 9.64, 16.00, 21.33, 27.97, 36.46, 47.76, 65.06, 88.64, 130.46, 220.66]

    # Enable/disable prediction projection
    USE_PROJECTION = False    # Whether to use prediction projection
    USE_SIGMOID = False       # Whether to use sigmoid activation in the regression head
    USE_TIME_SHIFT_AUGMENTATION = False  # Whether to use time shift augmentation
    USE_TIME_REVERSE_AUGMENTATION = False  # Whether to use time reverse augmentation
    SIGMOID_SCALE = 11.0      # Scale factor for sigmoid output 
    USE_CNNREGRESSOR = False      # Wheter to use Conformer or CNNRegressor

class PatchEmbedding(nn.Module):
    """
    CNN-based feature extraction module that replaces positional embeddings.
    Processes EEG data through temporal and spatial convolutions.
    """
    
    def __init__(self, emb_size: int = Config.EMB_SIZE, channels: int = Config.CNN_CHANNELS):
        super().__init__()

        self.shallownet = nn.Sequential(
            # Temporal convolution
            nn.Conv2d(1, channels, 
                     Config.TEMP_KERNEL_SIZE, 
                     Config.TEMP_STRIDE),
            # Spatial convolution
            nn.Conv2d(channels, channels, 
                     Config.SPATIAL_KERNEL_SIZE, 
                     Config.SPATIAL_STRIDE),
            nn.BatchNorm2d(channels),
            nn.ELU(),
            nn.AvgPool2d(Config.POOL_KERNEL_SIZE, 
                        Config.POOL_STRIDE),
            nn.Dropout(Config.DROPOUT),
        )

        self.projection = nn.Sequential(
            # 1x1 convolution for embedding transformation
            nn.Conv2d(channels, emb_size, (1, 1), stride=(1, 1)),
            # Flatten for Transformer input
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input EEG data into embeddings."""
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for capturing dependencies between sequence elements.
    """
    
    def __init__(self, 
                 emb_size: int = Config.EMB_SIZE, 
                 num_heads: int = Config.NUM_HEADS, 
                 dropout: float = Config.DROPOUT):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        
        # Linear projections for queries, keys, and values
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply multi-head attention mechanism."""
        # Split embeddings into multiple heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        # Compute attention scores
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        # Apply softmax and scaling
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)

        # Compute weighted sum of values
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)

        return out


class ResidualAdd(nn.Module):
    """Residual connection wrapper for adding input to output of a function."""
    
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply function and add result to input."""
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    """Feed-forward network used in Transformer blocks with expansion and dropout."""
    
    def __init__(self, 
                 emb_size: int = Config.EMB_SIZE, 
                 expansion: int = Config.FORWARD_EXPANSION, 
                 drop_p: float = Config.DROPOUT):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    """
    Transformer encoder block combining self-attention and feed-forward layers.
    Each component is wrapped with layer normalization and residual connections.
    """
    
    def __init__(self,
                 emb_size: int = Config.EMB_SIZE,
                 num_heads: int = Config.NUM_HEADS,
                 drop_p: float = Config.DROPOUT,
                 forward_expansion: int = Config.FORWARD_EXPANSION,
                 forward_drop_p: float = Config.DROPOUT):
        super().__init__(
            # Multi-head self-attention block
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            # Feed-forward block
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )


class TransformerEncoder(nn.Sequential):
    """Stack of Transformer encoder blocks."""
    
    def __init__(self, depth: int = Config.DEPTH, emb_size: int = Config.EMB_SIZE):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class RegressionHead(nn.Sequential):
    """
    Regression head that transforms sequence output into a continuous prediction.
    Uses global average pooling to aggregate sequence information.
    """
    
    def __init__(self, emb_size: int = Config.EMB_SIZE):
        super().__init__()
        
        # Define the head layers
        layers = [
            # Global average pooling across sequence dimension
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, 1)  # Output a single continuous value
        ]
        
        # Add sigmoid activation if configured
        if Config.USE_SIGMOID:
            layers.append(nn.Sigmoid())
            
        self.clshead = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through regression head."""
        # Handle different input shapes
        if len(x.shape) == 2:  
            x = x.unsqueeze(1) 
        elif len(x.shape) == 4:  
            x = x.view(x.size(0), x.size(1), -1).transpose(1, 2) 

        # Apply the head to get prediction
        out = self.clshead(x)
        
        # Scale the output if using sigmoid
        if Config.USE_SIGMOID:
            out = out * Config.SIGMOID_SCALE
            
        return out


class Conformer(nn.Sequential):
    """
    Conformer architecture combining CNN feature extraction with Transformer sequence modeling.
    """
    
    def __init__(self, 
                 emb_size: int = Config.EMB_SIZE, 
                 depth: int = Config.DEPTH, 
                 **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),            # CNN Feature Extraction
            TransformerEncoder(depth, emb_size), # Transformer for Sequence Modeling
            RegressionHead(emb_size)             # Regression Output
        ) 

class CNNRegressor(nn.Sequential):
    """
    EEG regression model using a CNN for feature extraction and a Regression Head for continuous value prediction.
    """

    def __init__(self, emb_size: int = Config.EMB_SIZE):
        super().__init__(
            PatchEmbedding(emb_size),  # CNN Feature Extraction
            RegressionHead(emb_size)   # Regression Output
        )


class EEGDataProcessor:
    """
    Handles loading and processing of EEG data from .mat files.
    Supports different data sources and formats.
    """

    @staticmethod
    def augment_eeg_data(eeg_data: np.ndarray) -> np.ndarray:
        """
        Augments EEG data by applying either time shift or time reverse augmentation.

        Args:
            eeg_data (np.ndarray): EEG data of shape (80000, 29).

        Returns:
            np.ndarray: Augmented EEG data containing original and augmented versions.
        """
        augmented_data = []

        # Keep original data
        original_eeg = np.transpose(eeg_data, (1, 0))  # Convert to (channels, time_points)
        original_eeg = np.expand_dims(original_eeg, axis=0)  # (1, channels, time_points)
        augmented_data.append(original_eeg)

        # Apply time reversal augmentation if enabled
        if Config.USE_TIME_REVERSE_AUGMENTATION:
            reversed_eeg = np.flip(original_eeg, axis=-1)  # Reverse time axis
            augmented_data.append(reversed_eeg)

        # Apply time shift augmentation if enabled
        if Config.USE_TIME_SHIFT_AUGMENTATION:
            num_segments = 2  # Total overlapping segments
            window_size = 60000  # Segment window size
            stride = 20000  # Step size for shifting

            for i in range(num_segments):
                start_idx = i * stride
                end_idx = start_idx + window_size

                if end_idx <= eeg_data.shape[-1]:  # Ensure valid indexing
                    segment = eeg_data[:, :, start_idx:end_idx]  # Shape: (1, channels, time_points)
                    augmented_data.append(segment)

        return np.array(augmented_data)  # Shape: (num_samples, channels, time_points)
    
    
    @staticmethod
    def load_continuous_labels(label_root: str) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Load continuous labels from .mat files and organize them by subject and scene IDs.
        
        Args:
            label_root: Path to the root directory containing label files
            
        Returns:
            Dictionary mapping (subject_id, scene_id) to label values
        """
        label_files = glob.glob(os.path.join(label_root, "train", "*.mat")) + \
                      glob.glob(os.path.join(label_root, "test", "*.mat"))
        label_dict = {}

        for file in label_files:
            mat_data = scipy.io.loadmat(file)
            label = mat_data['label'].astype(np.float32)

            match = re.search(r"subj_(\d+)_(\d+)", file)
            if match:
                subj_id = int(match.group(1))
                scene_id = int(match.group(2))
                label_dict[(subj_id, scene_id)] = label  

        return label_dict
    
    @staticmethod
    def load_direct_data(data_root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and process EEG data from .mat files where labels are stored with data.
        
        Args:
            data_root: Path to the root directory containing data files
            
        Returns:
            Tuple containing (train_data, train_labels, test_data, test_labels)
        """
        train_files = glob.glob(os.path.join(data_root, "train", "*.mat"))
        test_files = glob.glob(os.path.join(data_root, "test", "*.mat"))

        all_train_data = []
        all_train_labels = []
        all_test_data = []
        all_test_labels = []

        print(f"Number of train files: {len(train_files)} | Number of test files: {len(test_files)}", flush=True)

        expected_channels = Config.EEG_CHANNELS
        expected_time_points = Config.EEG_TIME_POINTS

        # Process training data
        for file in train_files:
            mat_data = scipy.io.loadmat(file)
            eeg_data = mat_data['data']  # Expected shape (time_points, channels)
            label = mat_data['label'].astype(np.float32)
            
            # Verify data shape or adapt to configured dimensions
            if eeg_data.shape != (expected_time_points, expected_channels):
                print(f"Warning: Data in {file} has shape {eeg_data.shape} but expected ({expected_time_points}, {expected_channels})", flush=True)
                
                # If number of channels doesn't match, we need to handle this case
                if eeg_data.shape[1] != expected_channels:
                    print(f"Error: Number of channels ({eeg_data.shape[1]}) doesn't match expected ({expected_channels})", flush=True)
                    continue

            if Config.USE_TIME_SHIFT_AUGMENTATION or Config.USE_TIME_REVERSE_AUGMENTATION:
                augmented_eeg_data = EEGDataProcessor.augment_eeg_data(eeg_data)  # (num_segments, channels, time_points)

                for segment in augmented_eeg_data:
                    all_train_data.append(segment)
                    all_train_labels.append(label)
            else:
                eeg_data = np.transpose(eeg_data, (1, 0))  # (channels, time_points)
                eeg_data = np.expand_dims(eeg_data, axis=0)  # (1, channels, time_points)

                all_train_data.append(eeg_data)
                all_train_labels.append(label)

        # Process test data
        for file in test_files:
            mat_data = scipy.io.loadmat(file)
            eeg_data = mat_data['data']
            label = mat_data['label'].astype(np.float32)
            
            # Verify data shape
            if eeg_data.shape != (expected_time_points, expected_channels):
                print(f"Warning: Data in {file} has shape {eeg_data.shape} but expected ({expected_time_points}, {expected_channels})", flush=True)
                
                # If number of channels doesn't match, we need to handle this case
                if eeg_data.shape[1] != expected_channels:
                    print(f"Error: Number of channels ({eeg_data.shape[1]}) doesn't match expected ({expected_channels})", flush=True)
                    continue

            eeg_data = np.transpose(eeg_data, (1, 0))
            eeg_data = np.expand_dims(eeg_data, axis=0)

            all_test_data.append(eeg_data)
            all_test_labels.append(label)

        # Convert lists to numpy arrays
        if not all_train_data:
            raise ValueError("No valid training data found")
        if not all_test_data:
            raise ValueError("No valid test data found")
            
        train_data = np.array(all_train_data)
        train_labels = np.array(all_train_labels).squeeze()

        test_data = np.array(all_test_data)
        test_labels = np.array(all_test_labels).squeeze()

        print(f"Training data shape: {train_data.shape}", flush=True)
        print(f"Test data shape: {test_data.shape}", flush=True)

        # Shuffle training data
        shuffle_idx = np.random.permutation(train_labels.shape[0])
        train_data = train_data[shuffle_idx]
        train_labels = train_labels[shuffle_idx]

        return train_data, train_labels, test_data, test_labels
    
    @staticmethod
    def load_separate_data(eeg_root: str, label_dict: Dict[Tuple[int, int], np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and process EEG data from .mat files where labels are stored separately.
        
        Args:
            eeg_root: Path to the root directory containing EEG data files
            label_dict: Dictionary mapping (subject_id, scene_id) to label values
            
        Returns:
            Tuple containing (train_data, train_labels, test_data, test_labels)
        """
        train_files = glob.glob(os.path.join(eeg_root, "train", "*mat"))
        test_files = glob.glob(os.path.join(eeg_root, "test", "*.mat"))

        all_train_data = []
        all_train_labels = []
        all_test_data = []
        all_test_labels = []

        print(f"Number of train files: {len(train_files)} | Number of test files: {len(test_files)}", flush=True)

        # Process training data
        for file in train_files:
            mat_data = scipy.io.loadmat(file)
            # Check if using augmented data (different key name)
            eeg_data = mat_data['data']

            match = re.search(r"subj_(\d+)_(\d+)", file)
            if match:
                subj_id, scene_id = int(match.group(1)), int(match.group(2))

                if (subj_id, scene_id) in label_dict:
                    label = label_dict[(subj_id, scene_id)]

                    if Config.USE_TIME_SHIFT_AUGMENTATION or Config.USE_TIME_REVERSE_AUGMENTATION:
                        augmented_eeg_data = EEGDataProcessor.augment_eeg_data(eeg_data)  # (num_segments, channels, time_points)

                        for segment in augmented_eeg_data:
                            all_train_data.append(segment)
                            all_train_labels.append(label)
                    else:
                        eeg_data = np.transpose(eeg_data, (1, 0))  
                        eeg_data = np.expand_dims(eeg_data, axis=0)  

                        all_train_data.append(eeg_data)
                        all_train_labels.append(label)

        # Process test data
        for file in test_files:
            mat_data = scipy.io.loadmat(file)
            # Apply the same key checking logic used for training files
            eeg_data = mat_data['data']
            
            match = re.search(r"subj_(\d+)_(\d+)", file)
            if match:
                subj_id, scene_id = int(match.group(1)), int(match.group(2))
                
                if (subj_id, scene_id) in label_dict:
                    label = label_dict[(subj_id, scene_id)]

                    eeg_data = np.transpose(eeg_data, (1, 0))
                    eeg_data = np.expand_dims(eeg_data, axis=0)  

                    all_test_data.append(eeg_data)
                    all_test_labels.append(label)

        # Convert lists to numpy arrays
        if not all_train_data:
            raise ValueError("No valid training data found")
        if not all_test_data:
            raise ValueError("No valid test data found")
            
        train_data = np.array(all_train_data) 
        train_labels = np.array(all_train_labels).squeeze() 

        test_data = np.array(all_test_data) 
        test_labels = np.array(all_test_labels).squeeze()

        print(f"Training data shape: {train_data.shape}", flush=True)
        print(f"Test data shape: {test_data.shape}", flush=True)

        # Shuffle training data
        shuffle_idx = np.random.permutation(train_labels.shape[0])
        train_data = train_data[shuffle_idx]
        train_labels = train_labels[shuffle_idx]

        return train_data, train_labels, test_data, test_labels


def project_prediction(predicted: torch.Tensor) -> torch.Tensor:
    """
    Project continuous predictions to specific value ranges based on discrete clusters.
    
    Args:
        predicted: Raw model predictions
        
    Returns:
        Projected predictions with values mapped to specific ranges
    """
    if not Config.USE_PROJECTION:
        return predicted
        
    predicted_detached = predicted.detach()
    cluster_int = torch.floor(predicted_detached).long()
    diff = predicted - torch.floor(predicted_detached)
    
    floors = torch.tensor(Config.RANGE_FLOORS,
                           device=predicted.device, dtype=predicted.dtype)
    uppers = torch.tensor(Config.RANGE_UPPERS,
                           device=predicted.device, dtype=predicted.dtype)
    
    indices = torch.clamp(cluster_int, 1, 11)
    
    floor_vals = floors[indices]
    upper_vals = uppers[indices]
    range_vals = upper_vals - floor_vals
    
    projected = floor_vals + diff * range_vals
    return projected


def set_seed(seed: Optional[int] = None) -> int:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed to use. If None, a random seed will be generated.
        
    Returns:
        The seed that was used.
    """
    if seed is None:
        seed = np.random.randint(2021)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    return seed 


class Experiment:
    """
    Main experiment class handling model training and evaluation.
    Supports different data loading methods and prediction projection.
    """
    
    def __init__(self, eeg_data_root: str, label_data_root: Optional[str] = None):
        """
        Initialize experiment with provided configuration.
        
        Args:
            eeg_data_root: Path to the EEG dataset root directory
            label_data_root: Optional path to separate label directory (if None, labels are assumed to be with data)
        """
        self.batch_size = Config.BATCH_SIZE
        self.n_epochs = Config.NUM_EPOCHS
        self.lr = Config.LEARNING_RATE
        self.b1 = Config.BETA1
        self.b2 = Config.BETA2
        self.eeg_data_root = eeg_data_root
        self.label_data_root = label_data_root
        self.use_separate_labels = label_data_root is not None

        # Setup CUDA tensors
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

        # Loss functions
        self.criterion_l1 = torch.nn.L1Loss().to(self.device)  # MAE
        self.criterion_l2 = torch.nn.MSELoss().to(self.device)  # MSE

        # Initialize model
        if Config.USE_CNNREGRESSOR:
            self.model = CNNRegressor().to(self.device)
        else:
            self.model = Conformer().to(self.device)

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model, device_ids=gpus)
            
        # Load labels if using separate label source
        if self.use_separate_labels:
            self.label_dict = EEGDataProcessor.load_continuous_labels(self.label_data_root)
        else:
            self.label_dict = None

    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training and testing data loaders.
        
        Returns:
            Tuple containing (train_dataloader, test_dataloader)
        """
        # Load data using the appropriate method
        if self.use_separate_labels:
            train_data, train_labels, test_data, test_labels = EEGDataProcessor.load_separate_data(
                self.eeg_data_root, self.label_dict)
        else:
            train_data, train_labels, test_data, test_labels = EEGDataProcessor.load_direct_data(
                self.eeg_data_root)
        
        # Convert to PyTorch tensors and create datasets
        train_data = torch.from_numpy(train_data).float()
        train_labels = torch.from_numpy(train_labels).float()
        train_dataset = TensorDataset(train_data, train_labels)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        test_data = torch.from_numpy(test_data).float()
        test_labels = torch.from_numpy(test_labels).float()
        test_dataset = TensorDataset(test_data, test_labels)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_loader, test_loader

    def train(self, job_id: str) -> Tuple[float, float]:
        """
        Train the model and evaluate performance.
        
        Args:
            job_id: Identifier for the current training job
            
        Returns:
            Tuple containing (best_mse, best_mae)
        """
        # Prepare data
        train_loader, test_loader = self.prepare_data()
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(self.b1, self.b2)
        )
        
        # Initialize tracking metrics
        best_mse = float('inf')
        best_mae = float('inf')
        
        train_mse_history = []
        train_mae_history = []
        test_mse_history = []
        test_mae_history = []
        
        # Create output directory for plots
        data_type = "original" if "eeg_original" in self.eeg_data_root else "clustered"
        if Config.USE_TIME_REVERSE_AUGMENTATION:
            data_type += "_time_reverse_augmented"
        elif Config.USE_TIME_SHIFT_AUGMENTATION:
            data_type += "_time_shift_augmented"
        
        exp_type = f"{data_type}_with_projection" if Config.USE_PROJECTION else f"{data_type}_no_projection"
        plot_dir = f"./output/conformer/{exp_type}/job_{job_id}/plots/"
        os.makedirs(plot_dir, exist_ok=True)
        
        # Training loop
        for epoch in range(self.n_epochs):
            # Training phase
            self.model.train()
            epoch_train_mae = 0
            epoch_train_mse = 0
            batch_count = 0
            
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_data)
                
                # Apply projection if enabled
                if Config.USE_PROJECTION:
                    projected_outputs = project_prediction(outputs.squeeze())
                else:
                    projected_outputs = outputs.squeeze()
                
                # Calculate losses
                loss_mse = self.criterion_l2(projected_outputs, batch_labels)
                loss_mae = self.criterion_l1(projected_outputs, batch_labels)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss_mae.backward()  # Using MAE for optimization
                self.optimizer.step()
                
                epoch_train_mae += loss_mae.item()
                epoch_train_mse += loss_mse.item()
                batch_count += 1
            
            # Calculate average losses for the epoch
            avg_train_mae = epoch_train_mae / batch_count
            avg_train_mse = epoch_train_mse / batch_count
            
            train_mse_history.append(avg_train_mse)
            train_mae_history.append(avg_train_mae)
            
            # Evaluation phase
            self.model.eval()
            with torch.no_grad():
                epoch_test_mse = 0
                epoch_test_mae = 0
                batch_count = 0
                
                for batch_data, batch_labels in test_loader:
                    batch_data = batch_data.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(batch_data)
                    
                    # Apply projection if enabled
                    if Config.USE_PROJECTION:
                        projected_outputs = project_prediction(outputs.squeeze())
                    else:
                        projected_outputs = outputs.squeeze()
                    
                    # Calculate losses
                    loss_test_mse = self.criterion_l2(projected_outputs, batch_labels)
                    loss_test_mae = self.criterion_l1(projected_outputs, batch_labels)
                    
                    epoch_test_mse += loss_test_mse.item()
                    epoch_test_mae += loss_test_mae.item()
                    batch_count += 1
                
                # Calculate average losses for the test set
                avg_test_mse = epoch_test_mse / batch_count
                avg_test_mae = epoch_test_mae / batch_count
                
                test_mse_history.append(avg_test_mse)
                test_mae_history.append(avg_test_mae)
                
                # Update best metrics
                if avg_test_mse < best_mse:
                    best_mse = avg_test_mse
                    # Save best model
                    model_path = f"./output/conformer/{exp_type}/job_{job_id}/best_model.pth"
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    torch.save(self.model.state_dict(), model_path)
                    
                if avg_test_mae < best_mae:
                    best_mae = avg_test_mae
                
                # Print progress
                print(f"\nEpoch {epoch}:", flush=True)
                print(f"Train - MSE: {avg_train_mse:.6f} | MAE: {avg_train_mae:.6f}", flush=True)
                print(f"Test  - MSE: {avg_test_mse:.6f} | MAE: {avg_test_mae:.6f}", flush=True)
                print(f"Best Test - MSE: {best_mse:.6f} | MAE: {best_mae:.6f}\n", flush=True)
        
        # Plot results
        self._save_loss_plots(
            train_mse_history, test_mse_history, 
            train_mae_history, test_mae_history, 
            plot_dir
        )
        
        return best_mse, best_mae
    
    def _save_loss_plots(self, 
                        train_mse: List[float], 
                        test_mse: List[float], 
                        train_mae: List[float], 
                        test_mae: List[float], 
                        plot_dir: str):
        """Save MSE and MAE loss plots to disk."""
        # Plot MSE Losses
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(train_mse)), train_mse, label="Train MSE", color="red")
        plt.plot(range(len(test_mse)), test_mse, label="Test MSE", color="orange")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.title("MSE Losses for EEG Data")
        plt.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.5)
        plt.legend()
        plt.savefig(f"{plot_dir}/mse_losses.png", dpi=300)
        plt.close()

        # Plot MAE Losses
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(train_mae)), train_mae, label="Train MAE", color="red")
        plt.plot(range(len(test_mae)), test_mae, label="Test MAE", color="orange")
        plt.xlabel("Epochs")
        plt.ylabel("MAE Loss")
        plt.title("MAE Losses for EEG Data")
        plt.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.5)
        plt.legend()
        plt.savefig(f"{plot_dir}/mae_losses.png", dpi=300)
        plt.close() 


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="EEG-based Regression with Conformer Architecture")
    
    # Data sources
    parser.add_argument(
        "--eeg_data_path", 
        type=str, 
        default=Config.DEFAULT_EEG_DATA_ROOT,
        help="Path to the EEG dataset root directory"
    )
    
    parser.add_argument(
        "--label_data_path", 
        type=str, 
        default=None,
        help="Path to the separate label dataset root directory (if needed)"
    )
    
    parser.add_argument(
        "--use_projection", 
        action="store_true",
        help="Enable prediction projection for discrete ranges"
    )
    
    parser.add_argument(
        "--use_sigmoid", 
        action="store_true",
        help="Enable sigmoid activation in regression head"
    )

    parser.add_argument(
        "--use_time_shift_augmentation", 
        action="store_true",
        help="Enable time shift augmentation"
    )

    parser.add_argument(
        "--use_time_reverse_augmentation", 
        action="store_true",
        help="Enable time reverse augmentation"
    )

    parser.add_argument(
        "--use_cnnregressor", 
        action="store_true",
        help="Enable CNNRegressor"
    )
    
    parser.add_argument(
        "--sigmoid_scale", 
        type=float, 
        default=Config.SIGMOID_SCALE,
        help="Scale factor for sigmoid output (if enabled)"
    )
    
    # Model parameters
    parser.add_argument(
        "--emb_size", 
        type=int, 
        default=Config.EMB_SIZE,
        help="Embedding size"
    )
    
    parser.add_argument(
        "--depth", 
        type=int, 
        default=Config.DEPTH,
        help="Transformer depth (number of blocks)"
    )
    
    parser.add_argument(
        "--num_heads", 
        type=int, 
        default=Config.NUM_HEADS,
        help="Number of attention heads"
    )
    
    parser.add_argument(
        "--dropout", 
        type=float, 
        default=Config.DROPOUT,
        help="Dropout rate"
    )
    
    # Training parameters
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=Config.BATCH_SIZE,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=Config.NUM_EPOCHS,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=Config.LEARNING_RATE,
        help="Learning rate"
    )
    
    # EEG data shape parameters
    parser.add_argument(
        "--eeg_channels", 
        type=int, 
        default=Config.EEG_CHANNELS,
        help="Number of EEG channels"
    )
    
    parser.add_argument(
        "--eeg_time_points", 
        type=int, 
        default=Config.EEG_TIME_POINTS,
        help="Number of time points in EEG data"
    )
    
    # Other parameters
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the experiment."""
    # Parse command line arguments
    args = parse_args()
    
    # Override config values with command line arguments
    Config.EMB_SIZE = args.emb_size
    Config.DEPTH = args.depth
    Config.NUM_HEADS = args.num_heads
    Config.DROPOUT = args.dropout
    Config.BATCH_SIZE = args.batch_size
    Config.NUM_EPOCHS = args.epochs
    Config.LEARNING_RATE = args.learning_rate
    Config.EEG_CHANNELS = args.eeg_channels
    Config.EEG_TIME_POINTS = args.eeg_time_points
    Config.USE_PROJECTION = args.use_projection
    Config.USE_SIGMOID = args.use_sigmoid
    Config.USE_TIME_SHIFT_AUGMENTATION = args.use_time_shift_augmentation
    Config.USE_TIME_REVERSE_AUGMENTATION = args.use_time_reverse_augmentation
    Config.SIGMOID_SCALE = args.sigmoid_scale
    Config.USE_CNNREGRESSOR = args.use_cnnregressor
    
    # Update dependent parameters
    Config.SPATIAL_KERNEL_SIZE = (Config.EEG_CHANNELS, 1)
    
    # Get job ID from environment or use local
    job_id = os.getenv("SLURM_JOB_ID", "local")
    
    print(f"Running job with ID: {job_id}", flush=True)
    print(f"Using EEG data path: {args.eeg_data_path}", flush=True)
    
    if args.label_data_path:
        print(f"Using separate label data path: {args.label_data_path}", flush=True)
    
    print(f"Model configuration:", flush=True)
    print(f"  - Embedding size: {Config.EMB_SIZE}", flush=True)
    print(f"  - Transformer depth: {Config.DEPTH}", flush=True)
    print(f"  - Number of attention heads: {Config.NUM_HEADS}", flush=True)
    
    print(f"EEG configuration: {Config.EEG_CHANNELS} channels, {Config.EEG_TIME_POINTS} time points", flush=True)
    
    print(f"Prediction projection: {'Enabled' if Config.USE_PROJECTION else 'Disabled'}", flush=True)
    print(f"Sigmoid in regression head: {'Enabled' if Config.USE_SIGMOID else 'Disabled'}, Scale: {Config.SIGMOID_SCALE}", flush=True)
    augmentation_status = (
        "Time Reverse Augmentation: Enabled" if Config.USE_TIME_REVERSE_AUGMENTATION else
        "Time Shift Augmentation: Enabled" if Config.USE_TIME_SHIFT_AUGMENTATION else
        "Data Augmentation: Disabled"
    )
    print(augmentation_status, flush=True)

    print(f"CNNRegressor: {'Enabled' if Config.USE_CNNREGRESSOR else 'Disabled'}", flush=True)
    
    start_time = datetime.datetime.now()

    # Set random seed
    seed = set_seed(args.seed)
    print(f'Seed: {seed}', flush=True)

    # Run experiment
    experiment = Experiment(
        eeg_data_root=args.eeg_data_path,
        label_data_root=args.label_data_path
    )
    best_mse, best_mae = experiment.train(job_id)

    # Report results
    end_time = datetime.datetime.now()
    print(f'Training duration: {str(end_time - start_time)}', flush=True)

    print(f'==============================', flush=True)
    print(f'FINAL RESULTS', flush=True)
    print(f'Best Test MSE: {best_mse:.6f}', flush=True)
    print(f'Best Test MAE: {best_mae:.6f}', flush=True)
    print(f'==============================', flush=True)


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())), flush=True)
    main()
    print(time.asctime(time.localtime(time.time())), flush=True) 
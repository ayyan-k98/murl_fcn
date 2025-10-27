"""
Spatial Softmax Module

Converts spatial feature maps to fixed-size coordinate vectors.
Grid-size invariant: Works on ANY H×W input.

Key Innovation:
- Input: [batch, channels, H, W] - ANY size
- Output: [batch, channels*2] - FIXED size (x,y coords per channel)

Used by: DeepMind, Google Brain for robotic manipulation.
Paper: "Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning"
       (Levine et al., 2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax: Convert spatial features to expected coordinates.

    For each feature channel, computes a "soft argmax" - the expected
    position (x,y) where that feature is most active.

    Example:
        Input: [2, 128, 20, 20] - 2 batches, 128 channels, 20×20 grid
        Output: [2, 256] - 128 channels × 2 coords (x,y) each

        Now try different grid size:
        Input: [2, 128, 50, 50] - Same batch/channels, 50×50 grid
        Output: [2, 256] - STILL 256! Grid-size invariant ✓

    Args:
        temperature: Softmax temperature (higher = more diffuse attention)
        normalized_coords: If True, use [-1, 1] coords; if False, use [0, 1]
    """

    def __init__(
        self,
        temperature: float = 1.0,
        normalized_coords: bool = True
    ):
        super().__init__()
        self.temperature = temperature
        self.normalized_coords = normalized_coords

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute expected coordinates for each feature channel.

        Args:
            features: [batch, channels, H, W] - ANY size!

        Returns:
            coords: [batch, channels*2] - Expected (x,y) for each channel
        """
        batch_size, num_channels, height, width = features.shape

        # Create coordinate grids
        if self.normalized_coords:
            # Normalized to [-1, 1] (standard for neural networks)
            y_coords = torch.linspace(-1, 1, height, device=features.device)
            x_coords = torch.linspace(-1, 1, width, device=features.device)
        else:
            # Normalized to [0, 1]
            y_coords = torch.linspace(0, 1, height, device=features.device)
            x_coords = torch.linspace(0, 1, width, device=features.device)

        # Expand coordinate grids to match batch and channel dimensions
        # y_grid: [batch, channels, H, W] - y coordinate at each position
        # x_grid: [batch, channels, H, W] - x coordinate at each position
        y_grid = y_coords.view(1, 1, height, 1).expand(batch_size, num_channels, height, width)
        x_grid = x_coords.view(1, 1, 1, width).expand(batch_size, num_channels, height, width)

        # Apply softmax to get attention weights
        # Reshape: [batch, channels, H*W]
        features_flat = features.view(batch_size, num_channels, -1)

        # Softmax over spatial dimension (H*W)
        # Higher feature values → higher attention weight
        attention = F.softmax(features_flat / self.temperature, dim=2)

        # Reshape back: [batch, channels, H, W]
        attention = attention.view(batch_size, num_channels, height, width)

        # Expected coordinates: weighted average over spatial positions
        # E[x] = Σ(attention * x_coord) over all positions
        # E[y] = Σ(attention * y_coord) over all positions
        expected_x = (attention * x_grid).sum(dim=[2, 3])  # [batch, channels]
        expected_y = (attention * y_grid).sum(dim=[2, 3])  # [batch, channels]

        # Interleave x,y coordinates: [x0, y0, x1, y1, ..., x127, y127]
        coords = torch.stack([expected_x, expected_y], dim=2)  # [batch, channels, 2]
        coords = coords.view(batch_size, -1)  # [batch, channels*2]

        return coords

    def visualize_attention(
        self,
        features: torch.Tensor,
        channel_idx: int = 0
    ) -> torch.Tensor:
        """
        Visualize attention map for a specific channel.

        Useful for debugging: Shows where each feature channel "looks".

        Args:
            features: [batch, channels, H, W]
            channel_idx: Which channel to visualize

        Returns:
            attention_map: [batch, H, W] - Attention weights for that channel
        """
        batch_size, num_channels, height, width = features.shape

        # Get attention weights for specified channel
        features_flat = features[:, channel_idx, :, :].view(batch_size, 1, -1)
        attention = F.softmax(features_flat / self.temperature, dim=2)
        attention_map = attention.view(batch_size, height, width)

        return attention_map


class SpatialSoftmaxWithTemperature(nn.Module):
    """
    Learnable temperature variant.

    Temperature is a learned parameter rather than fixed.
    Useful when optimal temperature is unknown.
    """

    def __init__(
        self,
        initial_temperature: float = 1.0,
        normalized_coords: bool = True
    ):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))
        self.normalized_coords = normalized_coords

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Same as SpatialSoftmax but with learnable temperature."""
        batch_size, num_channels, height, width = features.shape

        # Create coordinate grids
        if self.normalized_coords:
            y_coords = torch.linspace(-1, 1, height, device=features.device)
            x_coords = torch.linspace(-1, 1, width, device=features.device)
        else:
            y_coords = torch.linspace(0, 1, height, device=features.device)
            x_coords = torch.linspace(0, 1, width, device=features.device)

        y_grid = y_coords.view(1, 1, height, 1).expand(batch_size, num_channels, height, width)
        x_grid = x_coords.view(1, 1, 1, width).expand(batch_size, num_channels, height, width)

        # Softmax with learned temperature
        features_flat = features.view(batch_size, num_channels, -1)

        # Clamp temperature to avoid division by zero or extreme values
        temp = torch.clamp(self.temperature, min=0.1, max=10.0)
        attention = F.softmax(features_flat / temp, dim=2)
        attention = attention.view(batch_size, num_channels, height, width)

        # Expected coordinates
        expected_x = (attention * x_grid).sum(dim=[2, 3])
        expected_y = (attention * y_grid).sum(dim=[2, 3])

        coords = torch.stack([expected_x, expected_y], dim=2)
        coords = coords.view(batch_size, -1)

        return coords


# Test code
if __name__ == "__main__":
    print("="*70)
    print("TESTING SPATIAL SOFTMAX")
    print("="*70)

    # Test grid-size invariance
    spatial_softmax = SpatialSoftmax(temperature=1.0)

    # Test 1: Small grid (20×20)
    print("\nTest 1: Small grid (20×20)")
    features_small = torch.randn(2, 128, 20, 20)
    coords_small = spatial_softmax(features_small)
    print(f"  Input shape:  {features_small.shape}")
    print(f"  Output shape: {coords_small.shape}")
    print(f"  Expected:     torch.Size([2, 256])")
    print(f"  ✓ PASSED" if coords_small.shape == torch.Size([2, 256]) else "  ✗ FAILED")

    # Test 2: Large grid (50×50)
    print("\nTest 2: Large grid (50×50)")
    features_large = torch.randn(2, 128, 50, 50)
    coords_large = spatial_softmax(features_large)
    print(f"  Input shape:  {features_large.shape}")
    print(f"  Output shape: {coords_large.shape}")
    print(f"  Expected:     torch.Size([2, 256])")
    print(f"  ✓ PASSED" if coords_large.shape == torch.Size([2, 256]) else "  ✗ FAILED")

    # Test 3: Coordinate range
    print("\nTest 3: Coordinate range (should be [-1, 1])")
    print(f"  Min coord: {coords_large.min().item():.3f}")
    print(f"  Max coord: {coords_large.max().item():.3f}")
    in_range = (coords_large.min() >= -1.1) and (coords_large.max() <= 1.1)
    print(f"  ✓ PASSED" if in_range else "  ✗ FAILED")

    # Test 4: Gradient flow
    print("\nTest 4: Gradient flow")
    features = torch.randn(1, 64, 20, 20, requires_grad=True)
    coords = spatial_softmax(features)
    loss = coords.sum()
    loss.backward()
    has_grad = features.grad is not None and features.grad.abs().sum() > 0
    print(f"  Gradients present: {has_grad}")
    print(f"  ✓ PASSED" if has_grad else "  ✗ FAILED")

    # Test 5: Attention visualization
    print("\nTest 5: Attention visualization")
    features = torch.randn(1, 4, 10, 10)

    # Create a feature with clear peak at center
    features[0, 0, 5, 5] = 10.0  # Strong activation at center

    attention_map = spatial_softmax.visualize_attention(features, channel_idx=0)
    print(f"  Attention map shape: {attention_map.shape}")
    print(f"  Attention sum: {attention_map.sum().item():.3f} (should be ~1.0)")
    print(f"  Max attention location: {attention_map.argmax()}")
    print(f"  Expected location (center): {5*10 + 5}")
    print(f"  ✓ PASSED" if abs(attention_map.sum().item() - 1.0) < 0.01 else "  ✗ FAILED")

    # Test 6: Learnable temperature
    print("\nTest 6: Learnable temperature")
    spatial_softmax_learnable = SpatialSoftmaxWithTemperature(initial_temperature=2.0)
    features = torch.randn(2, 64, 15, 15)
    coords = spatial_softmax_learnable(features)
    print(f"  Initial temperature: {spatial_softmax_learnable.temperature.item():.3f}")
    print(f"  Output shape: {coords.shape}")
    print(f"  ✓ PASSED" if coords.shape == torch.Size([2, 128]) else "  ✗ FAILED")

    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)

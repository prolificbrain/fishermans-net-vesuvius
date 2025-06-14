#!/usr/bin/env python3
"""
Pure NumPy/SciPy implementation of the Fisherman's Net Volume Warping Algorithm
for the Vesuvius Challenge.

This implementation treats scroll deformation as a physics problem where:
- Fiber predictions act as "threads" we can pull
- Physics simulation ensures natural deformation
- Progressive unwrapping preserves papyrus integrity
"""

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class WarpingConfig:
    """Configuration for the warping algorithm."""
    elasticity: float = 0.8
    viscosity: float = 0.2
    fiber_strength: float = 2.0
    smoothing_sigma: float = 2.0
    max_deformation: float = 20.0
    convergence_threshold: float = 0.0001  # Stricter convergence
    num_critical_fibers: int = 100  # More fibers for better warping
    step_size: float = 0.2  # Larger steps for more visible deformation
    min_fiber_strength: float = 0.05  # Lower threshold to find more fibers


class FishermansNetWarperNumPy:
    """
    Pure NumPy implementation of the Fisherman's Net warping algorithm.
    
    This treats the scroll as a tangled fishing net where fiber predictions
    act as threads we can pull to unwrap the scroll naturally.
    """
    
    def __init__(self, config: WarpingConfig = None):
        self.config = config or WarpingConfig()
        self.deformation_history = []
        
    def warp_volume(self, 
                   volume: np.ndarray,
                   fiber_volume: np.ndarray,
                   fiber_orientations: np.ndarray,
                   mask: Optional[np.ndarray] = None,
                   num_iterations: int = 100) -> Dict:
        """
        Main warping function that progressively unwarps the scroll.
        
        Args:
            volume: 3D array of voxel intensities (z, y, x)
            fiber_volume: 3D array of fiber predictions (confidence scores)
            fiber_orientations: 4D array of fiber directions (z, y, x, 3)
            mask: Optional binary mask for valid regions
            num_iterations: Number of warping iterations
            
        Returns:
            Dictionary containing warped volume, deformation field, and metrics
        """
        logger.info(f"Starting warping process for volume of shape {volume.shape}")
        
        # Initialize deformation field
        shape = volume.shape
        deformation_field = np.zeros((*shape, 3), dtype=np.float32)
        velocity_field = np.zeros_like(deformation_field)
        
        # Find critical fiber paths - the "threads" to pull
        logger.info("Identifying critical fiber paths...")
        critical_fibers = self._find_critical_fibers(fiber_volume, fiber_orientations, mask)
        
        if len(critical_fibers) == 0:
            logger.warning("No critical fibers found! Creating some based on volume structure...")
            critical_fibers = self._create_structure_based_fibers(volume, mask)
        
        # Main warping loop
        metrics = {'flatness': [], 'strain': [], 'convergence': []}
        
        for iteration in range(num_iterations):
            # Compute forces that pull along fiber paths
            fiber_forces = self._compute_fiber_forces(
                deformation_field, critical_fibers, iteration / num_iterations
            )
            
            # Compute elastic forces to prevent over-deformation
            elastic_forces = self._compute_elastic_forces(deformation_field)
            
            # Update velocity and deformation
            total_forces = fiber_forces + elastic_forces
            velocity_field = (self.config.viscosity * velocity_field + 
                            self.config.step_size * total_forces)
            
            # Apply maximum deformation constraint
            velocity_magnitude = np.linalg.norm(velocity_field, axis=-1, keepdims=True)
            velocity_field = np.where(
                velocity_magnitude > self.config.max_deformation,
                velocity_field * self.config.max_deformation / (velocity_magnitude + 1e-10),
                velocity_field
            )
            
            deformation_field += velocity_field
            
            # Compute metrics every 10 iterations
            if iteration % 10 == 0:
                warped = self._apply_deformation(volume, deformation_field)
                flatness = self._compute_flatness_score(warped, mask)
                strain = self._compute_strain_energy(deformation_field)
                convergence = np.mean(np.abs(velocity_field))
                
                metrics['flatness'].append(flatness)
                metrics['strain'].append(strain)
                metrics['convergence'].append(convergence)
                
                logger.info(f"Iteration {iteration}: flatness={flatness:.4f}, "
                          f"strain={strain:.4f}, convergence={convergence:.6f}")
                
                # Check convergence - but only after some minimum iterations
                if iteration > 20 and convergence < self.config.convergence_threshold:
                    logger.info(f"Converged at iteration {iteration}")
                    break
        
        # Final warping
        warped_volume = self._apply_deformation(volume, deformation_field)
        
        return {
            'warped_volume': warped_volume,
            'deformation_field': deformation_field,
            'metrics': metrics,
            'critical_fibers': critical_fibers
        }
    
    def _find_critical_fibers(self, 
                             fiber_volume: np.ndarray,
                             fiber_orientations: np.ndarray,
                             mask: Optional[np.ndarray]) -> List[Dict]:
        """
        Find critical fiber paths that will guide the unwarping.
        These are the "threads" we'll pull to unwrap the scroll.
        """
        critical_fibers = []
        
        # Apply mask if provided
        if mask is not None:
            fiber_confidence = fiber_volume * mask
        else:
            fiber_confidence = fiber_volume
        
        # Find high-confidence fiber regions with lower threshold
        valid_fibers = fiber_confidence[fiber_confidence > 0]
        if len(valid_fibers) == 0:
            logger.warning("No fiber confidence found")
            return critical_fibers

        threshold = np.percentile(valid_fibers, 60)  # Lower threshold to find more fibers

        # Find fiber endpoints (high gradient regions)
        gradient = ndimage.gaussian_gradient_magnitude(fiber_confidence, sigma=1.0)

        # Use multiple threshold levels to find more endpoints
        gradient_thresholds = [np.percentile(gradient, p) for p in [85, 90, 95]]
        all_endpoints = []

        for grad_thresh in gradient_thresholds:
            endpoints = np.where(gradient > grad_thresh)
            if len(endpoints[0]) > 0:
                all_endpoints.extend(list(zip(endpoints[0], endpoints[1], endpoints[2])))

        # Remove duplicates and convert back to arrays
        unique_endpoints = list(set(all_endpoints))
        if not unique_endpoints:
            logger.warning("No fiber endpoints found")
            return critical_fibers

        endpoints = (np.array([p[0] for p in unique_endpoints]),
                    np.array([p[1] for p in unique_endpoints]),
                    np.array([p[2] for p in unique_endpoints]))
        
        if len(endpoints[0]) == 0:
            logger.warning("No fiber endpoints found")
            return critical_fibers
        
        # Sample critical points
        num_points = min(self.config.num_critical_fibers, len(endpoints[0]))
        indices = np.random.choice(len(endpoints[0]), num_points, replace=False)
        
        for idx in indices:
            start_point = np.array([endpoints[0][idx], endpoints[1][idx], endpoints[2][idx]])
            
            # Trace fiber from this point
            fiber_path = self._trace_fiber(start_point, fiber_volume, fiber_orientations)
            
            if len(fiber_path) > 5:  # Minimum path length
                # Compute average strength along path
                strengths = []
                for point in fiber_path:
                    z, y, x = point.astype(int)
                    if (0 <= z < fiber_volume.shape[0] and 
                        0 <= y < fiber_volume.shape[1] and 
                        0 <= x < fiber_volume.shape[2]):
                        strengths.append(fiber_volume[z, y, x])
                
                avg_strength = np.mean(strengths) if strengths else 0.1
                
                critical_fibers.append({
                    'path': fiber_path,
                    'strength': avg_strength,
                    'direction': self._compute_fiber_direction(fiber_path)
                })
        
        logger.info(f"Found {len(critical_fibers)} critical fibers")
        return critical_fibers
    
    def _create_structure_based_fibers(self, volume: np.ndarray, 
                                     mask: Optional[np.ndarray]) -> List[Dict]:
        """
        Create fiber paths based on volume structure when no fibers are detected.
        This uses the scroll's natural layered structure.
        """
        logger.info("Creating structure-based fibers from volume data...")
        critical_fibers = []
        
        # Find high-intensity regions (likely papyrus layers)
        threshold = np.percentile(volume, 85)
        structure_mask = volume > threshold
        
        if mask is not None:
            structure_mask = structure_mask & mask
        
        # Find connected components (individual papyrus layers)
        labeled, num_features = ndimage.label(structure_mask)
        
        # Create fibers along the major axis of each component
        for label_id in range(1, min(num_features + 1, self.config.num_critical_fibers + 1)):
            component_mask = labeled == label_id
            
            # Find the centroid and extent of this component
            coords = np.where(component_mask)
            if len(coords[0]) < 10:  # Skip very small components
                continue
            
            # Create a fiber path along the longest axis of the component
            z_coords, y_coords, x_coords = coords
            
            # Find the principal axis using PCA-like approach
            center = np.array([np.mean(z_coords), np.mean(y_coords), np.mean(x_coords)])
            
            # Create a path from one end to the other
            z_range = np.max(z_coords) - np.min(z_coords)
            y_range = np.max(y_coords) - np.min(y_coords)
            x_range = np.max(x_coords) - np.min(x_coords)
            
            # Choose the direction with the largest range
            if z_range >= y_range and z_range >= x_range:
                # Path along z-axis
                path_points = []
                for z in range(np.min(z_coords), np.max(z_coords) + 1, 2):
                    mask_slice = component_mask[z, :, :]
                    if np.any(mask_slice):
                        y_center, x_center = ndimage.center_of_mass(mask_slice)
                        path_points.append([z, y_center, x_center])
            elif y_range >= x_range:
                # Path along y-axis
                path_points = []
                for y in range(np.min(y_coords), np.max(y_coords) + 1, 2):
                    mask_slice = component_mask[:, y, :]
                    if np.any(mask_slice):
                        z_center, x_center = ndimage.center_of_mass(mask_slice)
                        path_points.append([z_center, y, x_center])
            else:
                # Path along x-axis
                path_points = []
                for x in range(np.min(x_coords), np.max(x_coords) + 1, 2):
                    mask_slice = component_mask[:, :, x]
                    if np.any(mask_slice):
                        z_center, y_center = ndimage.center_of_mass(mask_slice)
                        path_points.append([z_center, y_center, x])
            
            if len(path_points) > 5:
                fiber_path = np.array(path_points)
                
                # Compute average intensity along path as "strength"
                strengths = []
                for point in fiber_path:
                    z, y, x = point.astype(int)
                    if (0 <= z < volume.shape[0] and 
                        0 <= y < volume.shape[1] and 
                        0 <= x < volume.shape[2]):
                        strengths.append(volume[z, y, x])
                
                avg_strength = np.mean(strengths) / np.max(volume) if strengths else 0.1
                
                critical_fibers.append({
                    'path': fiber_path,
                    'strength': avg_strength,
                    'direction': self._compute_fiber_direction(fiber_path)
                })
        
        logger.info(f"Created {len(critical_fibers)} structure-based fibers")
        return critical_fibers

    def _trace_fiber(self, start_point: np.ndarray, fiber_volume: np.ndarray,
                    fiber_orientations: np.ndarray, max_steps: int = 100) -> np.ndarray:
        """Trace a single fiber path from starting point."""
        path = [start_point.copy()]
        current = start_point.astype(np.float32)

        for _ in range(max_steps):
            z, y, x = current.astype(int)

            # Check bounds
            if not (0 <= z < fiber_volume.shape[0] and
                    0 <= y < fiber_volume.shape[1] and
                    0 <= x < fiber_volume.shape[2]):
                break

            # Check fiber strength with configurable threshold
            if fiber_volume[z, y, x] < self.config.min_fiber_strength:
                break

            # Get fiber direction at current point
            direction = fiber_orientations[z, y, x]

            # Step along fiber
            next_point = current + direction * 0.5
            path.append(next_point.copy())
            current = next_point

        return np.array(path)

    def _compute_fiber_forces(self, deformation_field: np.ndarray,
                             critical_fibers: List[Dict], progress: float) -> np.ndarray:
        """Compute forces that pull along critical fibers."""
        forces = np.zeros_like(deformation_field)

        # Adaptive strength based on progress
        strength = self.config.fiber_strength * (1.0 - 0.7 * progress)

        for fiber in critical_fibers:
            path = fiber['path']
            direction = fiber['direction']
            fiber_strength = fiber['strength']

            # Apply forces along the fiber path
            for point in path:
                z, y, x = point.astype(int)

                if (0 <= z < forces.shape[0] and
                    0 <= y < forces.shape[1] and
                    0 <= x < forces.shape[2]):

                    # Force proportional to fiber strength
                    force_magnitude = strength * fiber_strength
                    forces[z, y, x] += direction * force_magnitude

        # Smooth forces
        for i in range(3):
            forces[..., i] = ndimage.gaussian_filter(forces[..., i],
                                                   sigma=self.config.smoothing_sigma)

        return forces

    def _compute_elastic_forces(self, deformation_field: np.ndarray) -> np.ndarray:
        """Compute elastic regularization forces."""
        elastic_forces = np.zeros_like(deformation_field)

        # Compute strain (simplified as gradient of deformation)
        for i in range(3):
            grad_z, grad_y, grad_x = np.gradient(deformation_field[..., i])
            elastic_forces[..., i] = -self.config.elasticity * (grad_z + grad_y + grad_x)

        return elastic_forces

    def _apply_deformation(self, volume: np.ndarray,
                          deformation_field: np.ndarray) -> np.ndarray:
        """Apply deformation field to volume using interpolation."""
        # Create coordinate grids
        z, y, x = np.mgrid[0:volume.shape[0], 0:volume.shape[1], 0:volume.shape[2]]

        # Apply deformation
        warped_coords_z = z + deformation_field[..., 0]
        warped_coords_y = y + deformation_field[..., 1]
        warped_coords_x = x + deformation_field[..., 2]

        # Clip coordinates to valid range
        warped_coords_z = np.clip(warped_coords_z, 0, volume.shape[0] - 1)
        warped_coords_y = np.clip(warped_coords_y, 0, volume.shape[1] - 1)
        warped_coords_x = np.clip(warped_coords_x, 0, volume.shape[2] - 1)

        # Use scipy's map_coordinates for interpolation
        coords = np.array([warped_coords_z.ravel(),
                          warped_coords_y.ravel(),
                          warped_coords_x.ravel()])

        warped_volume = ndimage.map_coordinates(volume, coords,
                                              order=1, mode='nearest')

        return warped_volume.reshape(volume.shape)

    def _compute_flatness_score(self, volume: np.ndarray,
                               mask: Optional[np.ndarray] = None) -> float:
        """Measure how flat the warped volume is."""
        if mask is not None:
            volume = volume * mask

        # Compute variation along z-axis (depth)
        z_std = np.std(volume, axis=0)

        # Average over valid regions
        if mask is not None:
            mask_2d = np.max(mask, axis=0)
            flatness = np.sum(z_std * mask_2d) / (np.sum(mask_2d) + 1e-10)
        else:
            flatness = np.mean(z_std)

        # Convert to score (higher is better)
        score = 1.0 / (1.0 + flatness)
        return float(score)

    def _compute_strain_energy(self, deformation_field: np.ndarray) -> float:
        """Compute total strain energy in the deformation."""
        strain_energy = 0.0
        for i in range(3):
            grad_z, grad_y, grad_x = np.gradient(deformation_field[..., i])
            strain_energy += np.sum(grad_z**2 + grad_y**2 + grad_x**2)
        return float(strain_energy)

    def _compute_fiber_direction(self, path: np.ndarray) -> np.ndarray:
        """Compute overall direction of a fiber path."""
        if len(path) < 2:
            return np.array([0.0, 0.0, 1.0])

        # Use endpoints to determine direction
        direction = path[-1] - path[0]
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction = direction / norm
        else:
            direction = np.array([0.0, 0.0, 1.0])

        return direction

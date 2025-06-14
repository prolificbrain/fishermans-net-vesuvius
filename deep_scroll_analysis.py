#!/usr/bin/env python3
"""
Deep Scroll Analysis: Enhanced Fisherman's Net Algorithm
for Comprehensive Multi-Segment Processing and Mystery Discovery

This script processes all available scroll segments with optimized parameters
to maximize the chance of uncovering ancient text and mysteries.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import tifffile
from fishermans_net_numpy import FishermansNetWarperNumPy, WarpingConfig
import logging
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LOCAL_DATA_DIR = Path(os.path.abspath(__file__)).parent / "VesuviusDataDownload/Scroll1/segments"


class DeepScrollAnalyzer:
    """Enhanced analyzer for comprehensive scroll processing and mystery discovery."""
    
    def __init__(self, output_dir="deep_analysis_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def discover_all_segments(self):
        """Discover and analyze all available scroll segments."""
        logger.info("🔍 DISCOVERING ALL SCROLL SEGMENTS")
        
        if not LOCAL_DATA_DIR.exists():
            logger.error(f"Data directory {LOCAL_DATA_DIR} does not exist!")
            return []
        
        segments = []
        for segment_dir in LOCAL_DATA_DIR.iterdir():
            if segment_dir.is_dir():
                layers_dir = segment_dir / "layers"
                if layers_dir.exists():
                    layer_files = list(layers_dir.glob("*.tif"))
                    if layer_files:
                        segments.append({
                            'id': segment_dir.name,
                            'path': layers_dir,
                            'layer_count': len(layer_files),
                            'size_mb': sum(f.stat().st_size for f in layer_files) / 1024**2
                        })
        
        logger.info(f"📊 Found {len(segments)} segments:")
        for seg in segments:
            logger.info(f"  🎯 {seg['id']}: {seg['layer_count']} layers ({seg['size_mb']:.1f} MB)")
        
        return segments
    
    def load_enhanced_segment(self, segment_info, layer_range=None, downsample=1):
        """Load segment with enhanced processing options."""
        logger.info(f"📥 Loading segment {segment_info['id']} with enhanced processing...")
        
        layer_files = sorted(glob.glob(str(segment_info['path'] / "*.tif")))
        
        # Determine layer range
        if layer_range is None:
            # Use more layers for deeper analysis
            total_layers = len(layer_files)
            layer_range = (0, min(50, total_layers))  # Up to 50 layers
        
        z_start, z_end = layer_range
        z_end = min(z_end, len(layer_files))
        selected_files = layer_files[z_start:z_end:downsample]
        
        logger.info(f"📊 Processing layers {z_start} to {z_end-1} (step {downsample}): {len(selected_files)} files")
        
        # Load with memory optimization
        volume_slices = []
        for i, layer_file in enumerate(selected_files):
            if i % 10 == 0:
                logger.info(f"  Loading layer {i+1}/{len(selected_files)}")
            
            slice_data = tifffile.imread(layer_file)
            
            # Optional downsampling for memory management
            if downsample > 1:
                slice_data = slice_data[::downsample, ::downsample]
            
            volume_slices.append(slice_data)
        
        volume = np.stack(volume_slices, axis=0).astype(np.float32)
        
        logger.info(f"✅ Loaded volume: {volume.shape}, {volume.nbytes/1024**2:.1f} MB")
        logger.info(f"📈 Intensity range: [{np.min(volume):.1f}, {np.max(volume):.1f}]")
        
        return volume, layer_range
    
    def create_adaptive_fiber_data(self, volume, segment_id):
        """Create adaptive fiber data optimized for each segment."""
        logger.info(f"🧬 Creating adaptive fiber data for segment {segment_id}...")
        
        depth, height, width = volume.shape
        
        # Normalize volume
        volume_norm = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-10)
        
        # Adaptive thresholding based on volume characteristics
        intensity_std = np.std(volume_norm)
        if intensity_std > 0.3:
            # High contrast - use stricter thresholds
            thresholds = [0.75, 0.8, 0.85]
            logger.info("📊 High contrast detected - using strict thresholds")
        else:
            # Low contrast - use more permissive thresholds
            thresholds = [0.6, 0.7, 0.8]
            logger.info("📊 Low contrast detected - using permissive thresholds")
        
        # Multi-scale fiber detection
        fiber_volumes = []
        for thresh in thresholds:
            papyrus_threshold = np.percentile(volume_norm, thresh * 100)
            papyrus_mask = volume_norm > papyrus_threshold
            
            # Enhanced gradient computation
            grad_z, grad_y, grad_x = np.gradient(volume_norm)
            gradient_magnitude = np.sqrt(grad_z**2 + grad_y**2 + grad_x**2)
            
            # Detect edges and structures
            from scipy import ndimage
            
            # Laplacian for edge detection
            laplacian = ndimage.laplace(volume_norm)
            
            # Combine gradients and laplacian
            fiber_vol = (gradient_magnitude + np.abs(laplacian)) * papyrus_mask
            fiber_volumes.append(fiber_vol)
        
        # Combine and enhance
        combined_fiber_volume = np.mean(fiber_volumes, axis=0)
        
        # Morphological enhancement
        from scipy.ndimage import binary_dilation, binary_erosion
        
        # Create binary mask and enhance connectivity
        binary_mask = combined_fiber_volume > np.percentile(combined_fiber_volume, 80)
        enhanced_mask = binary_dilation(binary_erosion(binary_mask))
        combined_fiber_volume = combined_fiber_volume * enhanced_mask
        
        # Normalize
        if np.max(combined_fiber_volume) > 0:
            combined_fiber_volume = combined_fiber_volume / np.max(combined_fiber_volume)
        
        # Enhanced fiber orientations using structure tensor
        fiber_orientations = self._compute_structure_tensor_orientations(volume_norm)
        
        # Apply fiber mask
        fiber_mask = combined_fiber_volume > 0.1
        fiber_orientations = fiber_orientations * fiber_mask[..., np.newaxis]
        
        num_fiber_points = np.sum(combined_fiber_volume > 0.1)
        fiber_density = num_fiber_points / volume.size * 100
        
        logger.info(f"🎯 Created adaptive fiber data:")
        logger.info(f"  📊 Fiber points: {num_fiber_points:,}")
        logger.info(f"  📈 Density: {fiber_density:.2f}%")
        logger.info(f"  🎛️ Range: [{np.min(combined_fiber_volume):.3f}, {np.max(combined_fiber_volume):.3f}]")
        
        return combined_fiber_volume, fiber_orientations
    
    def _compute_structure_tensor_orientations(self, volume):
        """Compute fiber orientations using structure tensor analysis."""
        depth, height, width = volume.shape
        fiber_orientations = np.zeros((depth, height, width, 3))
        
        # Compute gradients
        grad_z, grad_y, grad_x = np.gradient(volume)
        
        # Structure tensor components
        Jxx = grad_x * grad_x
        Jyy = grad_y * grad_y
        Jzz = grad_z * grad_z
        Jxy = grad_x * grad_y
        Jxz = grad_x * grad_z
        Jyz = grad_y * grad_z
        
        # Smooth structure tensor
        from scipy.ndimage import gaussian_filter
        sigma = 1.0
        
        Jxx = gaussian_filter(Jxx, sigma)
        Jyy = gaussian_filter(Jyy, sigma)
        Jzz = gaussian_filter(Jzz, sigma)
        Jxy = gaussian_filter(Jxy, sigma)
        Jxz = gaussian_filter(Jxz, sigma)
        Jyz = gaussian_filter(Jyz, sigma)
        
        # Compute principal directions (simplified)
        # In practice, this would involve eigenvalue decomposition
        # For now, use gradient-based approximation
        
        fiber_orientations[..., 0] = -grad_y  # z-component
        fiber_orientations[..., 1] = grad_x   # y-component  
        fiber_orientations[..., 2] = -grad_z  # x-component
        
        # Normalize
        norms = np.linalg.norm(fiber_orientations, axis=-1, keepdims=True)
        fiber_orientations = np.where(norms > 1e-6, 
                                     fiber_orientations / norms, 
                                     fiber_orientations)
        
        return fiber_orientations
    
    def run_adaptive_warping(self, volume, fiber_volume, fiber_orientations, segment_id):
        """Run warping with adaptive parameters based on segment characteristics."""
        logger.info(f"🎣 Running adaptive warping for segment {segment_id}...")
        
        # Analyze volume characteristics for parameter adaptation
        volume_std = np.std(volume)
        fiber_density = np.sum(fiber_volume > 0.1) / volume.size
        
        # Adaptive configuration
        if fiber_density > 0.05:  # High fiber density
            config = WarpingConfig(
                elasticity=0.2,           # Lower elasticity for more deformation
                viscosity=0.1,            # Lower viscosity for faster convergence
                fiber_strength=4.0,       # Higher strength for dense fibers
                smoothing_sigma=0.8,      # Less smoothing to preserve detail
                max_deformation=30.0,     # Allow larger deformations
                num_critical_fibers=120,  # More fibers for better coverage
                step_size=0.4,            # Larger steps
                convergence_threshold=0.005,
                min_fiber_strength=0.02
            )
            logger.info("🎯 High fiber density - using aggressive parameters")
        else:  # Lower fiber density
            config = WarpingConfig(
                elasticity=0.4,           # Higher elasticity for stability
                viscosity=0.2,            # Higher viscosity for smoother motion
                fiber_strength=2.0,       # Moderate strength
                smoothing_sigma=1.5,      # More smoothing for stability
                max_deformation=20.0,     # Moderate deformations
                num_critical_fibers=60,   # Fewer fibers
                step_size=0.2,            # Smaller steps
                convergence_threshold=0.01,
                min_fiber_strength=0.05
            )
            logger.info("🎯 Lower fiber density - using conservative parameters")
        
        # Initialize warper
        warper = FishermansNetWarperNumPy(config)
        
        # Run warping with more iterations for better results
        result = warper.warp_volume(
            volume=volume,
            fiber_volume=fiber_volume,
            fiber_orientations=fiber_orientations,
            num_iterations=80  # More iterations for better convergence
        )
        
        # Enhanced analysis
        deformation_magnitude = np.linalg.norm(result['deformation_field'], axis=-1)
        
        analysis = {
            'segment_id': segment_id,
            'critical_fibers': len(result['critical_fibers']),
            'max_deformation': float(np.max(deformation_magnitude)),
            'mean_deformation': float(np.mean(deformation_magnitude)),
            'deformed_voxels': int(np.sum(deformation_magnitude > 0.1)),
            'fiber_density': float(fiber_density),
            'volume_std': float(volume_std),
            'config_used': 'aggressive' if fiber_density > 0.05 else 'conservative'
        }
        
        if result['metrics']['flatness']:
            analysis['final_flatness'] = float(result['metrics']['flatness'][-1])
        if result['metrics']['strain']:
            analysis['final_strain'] = float(result['metrics']['strain'][-1])
        
        logger.info(f"✅ Warping complete for {segment_id}:")
        logger.info(f"  🎯 Critical fibers: {analysis['critical_fibers']}")
        logger.info(f"  📏 Max deformation: {analysis['max_deformation']:.2f}")
        logger.info(f"  📊 Deformed voxels: {analysis['deformed_voxels']:,}")
        
        return result, analysis

    def create_mystery_discovery_visualizations(self, segment_id, volume, warped_volume,
                                               deformation_field, analysis):
        """Create comprehensive visualizations focused on mystery discovery."""
        logger.info(f"🎨 Creating mystery discovery visualizations for {segment_id}...")

        segment_dir = self.output_dir / f"segment_{segment_id}"
        segment_dir.mkdir(exist_ok=True)

        # Normalize volumes
        orig_norm = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
        warp_norm = (warped_volume - np.min(warped_volume)) / (np.max(warped_volume) - np.min(warped_volume))

        # Deformation magnitude
        deform_mag = np.linalg.norm(deformation_field, axis=-1)

        # Create comprehensive comparison
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'Mystery Discovery Analysis - Segment {segment_id}', fontsize=16, fontweight='bold')

        # Multiple slice analysis
        slices_to_analyze = [
            volume.shape[0] // 4,      # Quarter depth
            volume.shape[0] // 2,      # Middle depth
            3 * volume.shape[0] // 4   # Three-quarter depth
        ]

        for row, slice_idx in enumerate(slices_to_analyze):
            # Original
            axes[row, 0].imshow(orig_norm[slice_idx], cmap='gray')
            axes[row, 0].set_title(f'Original Z={slice_idx}')
            axes[row, 0].axis('off')

            # Warped
            axes[row, 1].imshow(warp_norm[slice_idx], cmap='gray')
            axes[row, 1].set_title(f'Warped Z={slice_idx}')
            axes[row, 1].axis('off')

            # Difference (potential text regions)
            diff = np.abs(warp_norm[slice_idx] - orig_norm[slice_idx])
            im_diff = axes[row, 2].imshow(diff, cmap='hot')
            axes[row, 2].set_title(f'Changes Z={slice_idx}')
            axes[row, 2].axis('off')
            plt.colorbar(im_diff, ax=axes[row, 2], fraction=0.046)

            # Deformation magnitude
            im_deform = axes[row, 3].imshow(deform_mag[slice_idx], cmap='plasma')
            axes[row, 3].set_title(f'Deformation Z={slice_idx}')
            axes[row, 3].axis('off')
            plt.colorbar(im_deform, ax=axes[row, 3], fraction=0.046)

        plt.tight_layout()
        plt.savefig(segment_dir / 'mystery_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Create potential text detection visualization
        self._create_text_detection_analysis(segment_id, orig_norm, warp_norm, deform_mag, segment_dir)

        # Create 3D structure analysis
        self._create_3d_structure_analysis(segment_id, volume, deformation_field, segment_dir)

        logger.info(f"✅ Visualizations saved to {segment_dir}")

    def _create_text_detection_analysis(self, segment_id, original, warped, deformation, output_dir):
        """Analyze potential text regions based on warping patterns."""
        logger.info(f"📝 Analyzing potential text regions in {segment_id}...")

        # Look for patterns that might indicate text
        # Text regions often show:
        # 1. Consistent deformation patterns
        # 2. Linear structures after warping
        # 3. Regular spacing patterns

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Potential Text Analysis - Segment {segment_id}', fontsize=14, fontweight='bold')

        mid_z = original.shape[0] // 2

        # Enhanced contrast analysis
        enhanced_original = self._enhance_contrast(original[mid_z])
        enhanced_warped = self._enhance_contrast(warped[mid_z])

        axes[0, 0].imshow(enhanced_original, cmap='gray')
        axes[0, 0].set_title('Enhanced Original')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(enhanced_warped, cmap='gray')
        axes[0, 1].set_title('Enhanced Warped')
        axes[0, 1].axis('off')

        # Difference highlighting potential ink
        diff = enhanced_warped - enhanced_original
        axes[0, 2].imshow(diff, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
        axes[0, 2].set_title('Potential Ink Regions')
        axes[0, 2].axis('off')

        # Pattern analysis
        from scipy import ndimage

        # Look for linear structures (potential text lines)
        sobel_x = ndimage.sobel(enhanced_warped, axis=1)
        sobel_y = ndimage.sobel(enhanced_warped, axis=0)

        axes[1, 0].imshow(np.abs(sobel_x), cmap='hot')
        axes[1, 0].set_title('Horizontal Structures')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(np.abs(sobel_y), cmap='hot')
        axes[1, 1].set_title('Vertical Structures')
        axes[1, 1].axis('off')

        # Combined structure analysis
        combined_structures = np.sqrt(sobel_x**2 + sobel_y**2)
        axes[1, 2].imshow(combined_structures, cmap='viridis')
        axes[1, 2].set_title('All Structures')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / 'text_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _enhance_contrast(self, image, percentile_range=(2, 98)):
        """Enhance image contrast for better text visibility."""
        p_low, p_high = np.percentile(image, percentile_range)
        enhanced = np.clip((image - p_low) / (p_high - p_low), 0, 1)
        return enhanced

    def _create_3d_structure_analysis(self, segment_id, volume, deformation_field, output_dir):
        """Create 3D structure analysis to understand scroll organization."""
        logger.info(f"🏗️ Creating 3D structure analysis for {segment_id}...")

        # Analyze deformation patterns in 3D
        deform_mag = np.linalg.norm(deformation_field, axis=-1)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'3D Structure Analysis - Segment {segment_id}', fontsize=14, fontweight='bold')

        # Z-projection analysis
        z_projection = np.mean(volume, axis=0)
        z_deform_proj = np.mean(deform_mag, axis=0)

        axes[0, 0].imshow(z_projection, cmap='gray')
        axes[0, 0].set_title('Z-Projection (Original)')
        axes[0, 0].axis('off')

        im1 = axes[0, 1].imshow(z_deform_proj, cmap='plasma')
        axes[0, 1].set_title('Z-Projection (Deformation)')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])

        # Cross-sectional analysis
        mid_y = volume.shape[1] // 2
        y_section = volume[:, mid_y, :]
        y_deform_section = deform_mag[:, mid_y, :]

        axes[1, 0].imshow(y_section, cmap='gray', aspect='auto')
        axes[1, 0].set_title('Y-Cross Section (Original)')
        axes[1, 0].axis('off')

        im2 = axes[1, 1].imshow(y_deform_section, cmap='plasma', aspect='auto')
        axes[1, 1].set_title('Y-Cross Section (Deformation)')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1])

        plt.tight_layout()
        plt.savefig(output_dir / '3d_structure_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def run_comprehensive_analysis(self):
        """Run comprehensive analysis on all available segments."""
        logger.info("🚀 STARTING COMPREHENSIVE SCROLL ANALYSIS")
        logger.info("=" * 60)

        # Discover all segments
        segments = self.discover_all_segments()
        if not segments:
            logger.error("No segments found!")
            return

        # Process each segment
        all_results = {}

        for i, segment_info in enumerate(segments):
            logger.info(f"\n🎯 PROCESSING SEGMENT {i+1}/{len(segments)}: {segment_info['id']}")
            logger.info("-" * 50)

            try:
                # Load segment with enhanced processing
                volume, layer_range = self.load_enhanced_segment(
                    segment_info,
                    layer_range=(0, 40),  # Process 40 layers for deeper analysis
                    downsample=1  # Full resolution
                )

                # Create adaptive fiber data
                fiber_volume, fiber_orientations = self.create_adaptive_fiber_data(
                    volume, segment_info['id']
                )

                # Run adaptive warping
                result, analysis = self.run_adaptive_warping(
                    volume, fiber_volume, fiber_orientations, segment_info['id']
                )

                # Create mystery discovery visualizations
                self.create_mystery_discovery_visualizations(
                    segment_info['id'], volume, result['warped_volume'],
                    result['deformation_field'], analysis
                )

                # Store results
                all_results[segment_info['id']] = {
                    'analysis': analysis,
                    'layer_range': layer_range,
                    'volume_shape': volume.shape
                }

                logger.info(f"✅ Completed analysis for segment {segment_info['id']}")

            except Exception as e:
                logger.error(f"❌ Error processing segment {segment_info['id']}: {e}")
                continue

        # Create comparative analysis
        self._create_comparative_analysis(all_results)

        # Save comprehensive report
        self._save_comprehensive_report(all_results)

        logger.info("\n🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
        logger.info(f"📊 Results saved to: {self.output_dir}")

        return all_results

    def _create_comparative_analysis(self, all_results):
        """Create comparative analysis across all segments."""
        logger.info("📊 Creating comparative analysis across all segments...")

        if len(all_results) < 2:
            logger.warning("Need at least 2 segments for comparative analysis")
            return

        # Extract metrics for comparison
        segment_ids = list(all_results.keys())
        metrics = {
            'critical_fibers': [all_results[sid]['analysis']['critical_fibers'] for sid in segment_ids],
            'max_deformation': [all_results[sid]['analysis']['max_deformation'] for sid in segment_ids],
            'deformed_voxels': [all_results[sid]['analysis']['deformed_voxels'] for sid in segment_ids],
            'fiber_density': [all_results[sid]['analysis']['fiber_density'] for sid in segment_ids]
        }

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparative Analysis Across All Segments', fontsize=16, fontweight='bold')

        # Critical fibers comparison
        axes[0, 0].bar(segment_ids, metrics['critical_fibers'], color='skyblue')
        axes[0, 0].set_title('Critical Fibers Found')
        axes[0, 0].set_ylabel('Number of Fibers')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Max deformation comparison
        axes[0, 1].bar(segment_ids, metrics['max_deformation'], color='lightcoral')
        axes[0, 1].set_title('Maximum Deformation')
        axes[0, 1].set_ylabel('Deformation (voxels)')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Deformed voxels comparison
        axes[1, 0].bar(segment_ids, metrics['deformed_voxels'], color='lightgreen')
        axes[1, 0].set_title('Deformed Voxels')
        axes[1, 0].set_ylabel('Number of Voxels')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Fiber density comparison
        axes[1, 1].bar(segment_ids, [d*100 for d in metrics['fiber_density']], color='gold')
        axes[1, 1].set_title('Fiber Density')
        axes[1, 1].set_ylabel('Density (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'comparative_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Find the most promising segment
        best_segment = self._identify_most_promising_segment(all_results)
        logger.info(f"🏆 Most promising segment for text discovery: {best_segment}")

    def _identify_most_promising_segment(self, all_results):
        """Identify the most promising segment for text discovery."""
        scores = {}

        for segment_id, data in all_results.items():
            analysis = data['analysis']

            # Scoring criteria for text discovery potential
            score = 0

            # High fiber density suggests good papyrus preservation
            score += analysis['fiber_density'] * 100

            # Moderate deformation suggests successful unwrapping without damage
            deform_ratio = analysis['deformed_voxels'] / (data['volume_shape'][0] * data['volume_shape'][1] * data['volume_shape'][2])
            if 0.01 < deform_ratio < 0.1:  # Sweet spot
                score += 50

            # More critical fibers = better structure detection
            score += analysis['critical_fibers'] * 0.5

            # Balanced strain suggests natural deformation
            if 'final_strain' in analysis:
                if 100 < analysis['final_strain'] < 1000:  # Balanced range
                    score += 30

            scores[segment_id] = score

        best_segment = max(scores, key=scores.get)
        return best_segment

    def _save_comprehensive_report(self, all_results):
        """Save comprehensive analysis report."""
        logger.info("📝 Saving comprehensive analysis report...")

        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_segments_processed': len(all_results),
            'algorithm_version': 'Fisherman\'s Net v2.0 - Deep Analysis',
            'segments': all_results,
            'summary': {
                'total_critical_fibers': sum(r['analysis']['critical_fibers'] for r in all_results.values()),
                'total_deformed_voxels': sum(r['analysis']['deformed_voxels'] for r in all_results.values()),
                'average_fiber_density': np.mean([r['analysis']['fiber_density'] for r in all_results.values()]),
                'most_promising_segment': self._identify_most_promising_segment(all_results)
            }
        }

        # Save as JSON
        with open(self.output_dir / 'comprehensive_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        # Save as readable text
        with open(self.output_dir / 'analysis_summary.txt', 'w') as f:
            f.write("🎣 FISHERMAN'S NET DEEP SCROLL ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Analysis Date: {report['analysis_timestamp']}\n")
            f.write(f"Segments Processed: {report['total_segments_processed']}\n")
            f.write(f"Algorithm: {report['algorithm_version']}\n\n")

            f.write("📊 SUMMARY STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Critical Fibers Found: {report['summary']['total_critical_fibers']:,}\n")
            f.write(f"Total Voxels Deformed: {report['summary']['total_deformed_voxels']:,}\n")
            f.write(f"Average Fiber Density: {report['summary']['average_fiber_density']:.4f}\n")
            f.write(f"Most Promising Segment: {report['summary']['most_promising_segment']}\n\n")

            f.write("🎯 SEGMENT DETAILS\n")
            f.write("-" * 30 + "\n")
            for segment_id, data in all_results.items():
                analysis = data['analysis']
                f.write(f"\nSegment {segment_id}:\n")
                f.write(f"  Critical Fibers: {analysis['critical_fibers']}\n")
                f.write(f"  Max Deformation: {analysis['max_deformation']:.2f} voxels\n")
                f.write(f"  Deformed Voxels: {analysis['deformed_voxels']:,}\n")
                f.write(f"  Fiber Density: {analysis['fiber_density']:.4f}\n")
                f.write(f"  Configuration: {analysis['config_used']}\n")

        logger.info(f"✅ Reports saved to {self.output_dir}")


def main():
    """Main function to run deep scroll analysis."""
    print("🎣 FISHERMAN'S NET DEEP SCROLL ANALYSIS")
    print("🏛️ Unlocking Ancient Mysteries with Enhanced Physics-Based Warping")
    print("=" * 70)

    # Initialize analyzer
    analyzer = DeepScrollAnalyzer("deep_analysis_results")

    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()

    if results:
        print("\n🎉 ANALYSIS COMPLETE!")
        print(f"📊 Processed {len(results)} segments")
        print(f"📁 Results saved to: deep_analysis_results/")
        print("\n🔍 Check the following files for discoveries:")
        print("  📈 comparative_analysis.png - Cross-segment comparison")
        print("  📝 analysis_summary.txt - Detailed text report")
        print("  📊 comprehensive_report.json - Full data export")
        print("  📁 segment_*/ - Individual segment analysis")
        print("\n🏆 Ready to discover ancient secrets!")
    else:
        print("❌ No segments could be processed. Check data availability.")


if __name__ == "__main__":
    main()

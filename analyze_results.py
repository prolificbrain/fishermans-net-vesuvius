#!/usr/bin/env python3
"""
Analysis script to understand the Fisherman's Net warping results
and create a comprehensive report for the Vesuvius Challenge submission.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import tifffile
from fishermans_net_numpy import FishermansNetWarperNumPy, WarpingConfig
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOCAL_DATA_DIR = Path(os.path.abspath(__file__)).parent / "VesuviusDataDownload/Scroll1/segments"


def load_and_analyze_scroll_segment(segment_id="20230518012543", layers=(0, 30)):
    """Load and perform detailed analysis of a scroll segment."""
    logger.info(f"=== ANALYZING SCROLL SEGMENT {segment_id} ===")
    
    segment_path = LOCAL_DATA_DIR / str(segment_id) / "layers"
    layer_files = sorted(glob.glob(str(segment_path / "*.tif")))
    
    z_start, z_end = layers
    z_end = min(z_end, len(layer_files))
    selected_files = layer_files[z_start:z_end]
    
    # Load data
    volume_slices = []
    for layer_file in selected_files:
        slice_data = tifffile.imread(layer_file)
        volume_slices.append(slice_data)
    
    volume = np.stack(volume_slices, axis=0).astype(np.float32)
    
    # Analyze volume characteristics
    logger.info(f"Volume shape: {volume.shape}")
    logger.info(f"Volume size: {volume.size:,} voxels")
    logger.info(f"Memory usage: {volume.nbytes / 1024**2:.1f} MB")
    logger.info(f"Value range: [{np.min(volume):.1f}, {np.max(volume):.1f}]")
    logger.info(f"Mean intensity: {np.mean(volume):.1f}")
    logger.info(f"Std intensity: {np.std(volume):.1f}")
    
    # Analyze intensity distribution
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    perc_values = np.percentile(volume, percentiles)
    for p, v in zip(percentiles, perc_values):
        logger.info(f"{p}th percentile: {v:.1f}")
    
    return volume


def create_enhanced_fiber_data(volume):
    """Create enhanced fiber data with detailed analysis."""
    logger.info("=== CREATING ENHANCED FIBER DATA ===")
    
    depth, height, width = volume.shape
    
    # Normalize volume
    volume_norm = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-10)
    
    # Multi-threshold approach for better fiber detection
    thresholds = [0.6, 0.7, 0.8, 0.85]
    fiber_volumes = []
    
    for thresh in thresholds:
        papyrus_threshold = np.percentile(volume_norm, thresh * 100)
        papyrus_mask = volume_norm > papyrus_threshold
        
        # Create fiber volume based on edges
        fiber_vol = np.zeros_like(volume_norm)
        
        # Compute gradients in all directions
        grad_z, grad_y, grad_x = np.gradient(volume_norm)
        gradient_magnitude = np.sqrt(grad_z**2 + grad_y**2 + grad_x**2)
        
        # Enhance areas with high gradients (edges) within papyrus regions
        fiber_vol = gradient_magnitude * papyrus_mask
        fiber_volumes.append(fiber_vol)
    
    # Combine fiber volumes
    combined_fiber_volume = np.mean(fiber_volumes, axis=0)
    
    # Normalize
    if np.max(combined_fiber_volume) > 0:
        combined_fiber_volume = combined_fiber_volume / np.max(combined_fiber_volume)
    
    # Create more sophisticated fiber orientations
    fiber_orientations = np.zeros((depth, height, width, 3))
    
    # Use structure tensor for better orientation estimation
    grad_z, grad_y, grad_x = np.gradient(volume_norm)
    
    # Fiber orientations based on local structure
    # In scrolls, fibers tend to follow the papyrus layers
    fiber_orientations[..., 0] = -grad_y  # z-component
    fiber_orientations[..., 1] = grad_x   # y-component  
    fiber_orientations[..., 2] = -grad_z  # x-component
    
    # Normalize orientation vectors
    norms = np.linalg.norm(fiber_orientations, axis=-1, keepdims=True)
    fiber_orientations = np.where(norms > 1e-6, 
                                 fiber_orientations / norms, 
                                 fiber_orientations)
    
    # Apply fiber mask
    fiber_mask = combined_fiber_volume > 0.1
    fiber_orientations = fiber_orientations * fiber_mask[..., np.newaxis]
    
    num_fiber_points = np.sum(combined_fiber_volume > 0.1)
    logger.info(f"Created fiber data with {num_fiber_points:,} fiber points")
    logger.info(f"Fiber density: {num_fiber_points / volume.size * 100:.2f}%")
    logger.info(f"Fiber volume range: [{np.min(combined_fiber_volume):.3f}, {np.max(combined_fiber_volume):.3f}]")
    
    return combined_fiber_volume, fiber_orientations


def run_comprehensive_warping_analysis(volume, fiber_volume, fiber_orientations):
    """Run comprehensive warping analysis with multiple configurations."""
    logger.info("=== RUNNING COMPREHENSIVE WARPING ANALYSIS ===")
    
    results = {}
    
    # Test different configurations
    configs = {
        'conservative': WarpingConfig(
            elasticity=0.8, viscosity=0.5, fiber_strength=1.0,
            max_deformation=10.0, num_critical_fibers=30,
            step_size=0.1, convergence_threshold=0.001
        ),
        'aggressive': WarpingConfig(
            elasticity=0.2, viscosity=0.1, fiber_strength=5.0,
            max_deformation=50.0, num_critical_fibers=100,
            step_size=0.5, convergence_threshold=0.01
        ),
        'balanced': WarpingConfig(
            elasticity=0.4, viscosity=0.2, fiber_strength=2.5,
            max_deformation=25.0, num_critical_fibers=60,
            step_size=0.3, convergence_threshold=0.005
        )
    }
    
    for config_name, config in configs.items():
        logger.info(f"--- Testing {config_name.upper()} configuration ---")
        
        warper = FishermansNetWarperNumPy(config)
        
        result = warper.warp_volume(
            volume=volume,
            fiber_volume=fiber_volume,
            fiber_orientations=fiber_orientations,
            num_iterations=50
        )
        
        results[config_name] = result
        
        # Analyze results
        logger.info(f"Critical fibers found: {len(result['critical_fibers'])}")
        if result['metrics']['flatness']:
            logger.info(f"Final flatness: {result['metrics']['flatness'][-1]:.6f}")
        if result['metrics']['strain']:
            logger.info(f"Final strain: {result['metrics']['strain'][-1]:.6f}")
        if result['metrics']['convergence']:
            logger.info(f"Final convergence: {result['metrics']['convergence'][-1]:.6f}")
        
        # Compute additional metrics
        deformation_magnitude = np.linalg.norm(result['deformation_field'], axis=-1)
        logger.info(f"Max deformation: {np.max(deformation_magnitude):.2f}")
        logger.info(f"Mean deformation: {np.mean(deformation_magnitude):.2f}")
        logger.info(f"Deformed voxels: {np.sum(deformation_magnitude > 0.1):,}")
        
        print()  # Add spacing
    
    return results


def create_comprehensive_visualizations(volume, results, output_dir):
    """Create comprehensive visualizations comparing all results."""
    logger.info("=== CREATING COMPREHENSIVE VISUALIZATIONS ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Normalize original volume
    orig_norm = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    mid_z = volume.shape[0] // 2
    
    # Original volume
    axes[0, 0].imshow(orig_norm[mid_z], cmap='gray')
    axes[0, 0].set_title('Original Volume')
    axes[0, 0].axis('off')
    
    # Results for each configuration
    for i, (config_name, result) in enumerate(results.items()):
        col = i + 1
        
        # Warped volume
        warped = result['warped_volume']
        warped_norm = (warped - np.min(warped)) / (np.max(warped) - np.min(warped))
        
        axes[0, col].imshow(warped_norm[mid_z], cmap='gray')
        axes[0, col].set_title(f'Warped ({config_name.title()})')
        axes[0, col].axis('off')
        
        # Deformation magnitude
        deform_mag = np.linalg.norm(result['deformation_field'], axis=-1)
        im = axes[1, col].imshow(deform_mag[mid_z], cmap='hot')
        axes[1, col].set_title(f'Deformation ({config_name.title()})')
        axes[1, col].axis('off')
        plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
    
    # Difference from original
    axes[1, 0].text(0.5, 0.5, 'Fisherman\'s Net\nVolume Warping\n\nfor Vesuvius Challenge', 
                   ha='center', va='center', transform=axes[1, 0].transAxes,
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[1, 0].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create metrics comparison
    plt.figure(figsize=(15, 5))
    
    config_names = list(results.keys())
    
    # Flatness comparison
    plt.subplot(1, 3, 1)
    flatness_scores = [results[name]['metrics']['flatness'][-1] if results[name]['metrics']['flatness'] 
                      else 0 for name in config_names]
    plt.bar(config_names, flatness_scores)
    plt.title('Flatness Score (Higher = Better)')
    plt.ylabel('Flatness')
    
    # Strain comparison
    plt.subplot(1, 3, 2)
    strain_scores = [results[name]['metrics']['strain'][-1] if results[name]['metrics']['strain'] 
                    else 0 for name in config_names]
    plt.bar(config_names, strain_scores)
    plt.title('Strain Energy')
    plt.ylabel('Strain')
    
    # Critical fibers comparison
    plt.subplot(1, 3, 3)
    fiber_counts = [len(results[name]['critical_fibers']) for name in config_names]
    plt.bar(config_names, fiber_counts)
    plt.title('Critical Fibers Found')
    plt.ylabel('Number of Fibers')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()


def generate_submission_report(volume, results, output_dir):
    """Generate a comprehensive report for Vesuvius Challenge submission."""
    logger.info("=== GENERATING SUBMISSION REPORT ===")
    
    report_path = os.path.join(output_dir, 'fishermans_net_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Fisherman's Net Volume Warping for Vesuvius Challenge\n\n")
        f.write("## Algorithm Overview\n\n")
        f.write("The Fisherman's Net algorithm treats ancient scroll deformation as a physics problem where:\n")
        f.write("- Fiber predictions act as 'threads' we can pull\n")
        f.write("- Physics simulation ensures natural deformation\n")
        f.write("- Progressive unwrapping preserves papyrus integrity\n\n")
        
        f.write("## Data Analysis\n\n")
        f.write(f"- **Volume Shape**: {volume.shape}\n")
        f.write(f"- **Total Voxels**: {volume.size:,}\n")
        f.write(f"- **Value Range**: [{np.min(volume):.1f}, {np.max(volume):.1f}]\n")
        f.write(f"- **Mean Intensity**: {np.mean(volume):.1f}\n\n")
        
        f.write("## Results Summary\n\n")
        
        for config_name, result in results.items():
            f.write(f"### {config_name.title()} Configuration\n\n")
            f.write(f"- **Critical Fibers Found**: {len(result['critical_fibers'])}\n")
            
            if result['metrics']['flatness']:
                f.write(f"- **Final Flatness Score**: {result['metrics']['flatness'][-1]:.6f}\n")
            if result['metrics']['strain']:
                f.write(f"- **Final Strain Energy**: {result['metrics']['strain'][-1]:.6f}\n")
            
            deform_mag = np.linalg.norm(result['deformation_field'], axis=-1)
            f.write(f"- **Max Deformation**: {np.max(deform_mag):.2f} voxels\n")
            f.write(f"- **Mean Deformation**: {np.mean(deform_mag):.2f} voxels\n")
            f.write(f"- **Deformed Voxels**: {np.sum(deform_mag > 0.1):,}\n\n")
        
        f.write("## Key Innovation\n\n")
        f.write("The Fisherman's Net approach is unique because it:\n")
        f.write("1. Uses fiber predictions as physical constraints\n")
        f.write("2. Applies progressive deformation rather than rigid unwrapping\n")
        f.write("3. Preserves local papyrus structure while correcting global distortion\n")
        f.write("4. Scales efficiently to large volumes using NumPy/SciPy\n\n")
        
        f.write("## Next Steps\n\n")
        f.write("- Integrate real fiber predictions from segmentation models\n")
        f.write("- Optimize parameters for different scroll types\n")
        f.write("- Apply to full scroll volumes\n")
        f.write("- Validate results with ground truth data\n")
    
    logger.info(f"Report saved to: {report_path}")


def main():
    """Run comprehensive analysis and generate submission materials."""
    output_dir = Path("./comprehensive_analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    # Load and analyze scroll data
    volume = load_and_analyze_scroll_segment(layers=(0, 25))
    
    # Create enhanced fiber data
    fiber_volume, fiber_orientations = create_enhanced_fiber_data(volume)
    
    # Run comprehensive warping analysis
    results = run_comprehensive_warping_analysis(volume, fiber_volume, fiber_orientations)
    
    # Create visualizations
    create_comprehensive_visualizations(volume, results, str(output_dir))
    
    # Generate submission report
    generate_submission_report(volume, results, str(output_dir))
    
    logger.info("=== ANALYSIS COMPLETE ===")
    logger.info(f"All results saved to: {output_dir}")
    logger.info("Ready for Vesuvius Challenge submission!")


if __name__ == "__main__":
    main()

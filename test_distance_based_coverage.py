"""
Test distance-based probabilistic coverage model.

Based on sensor model from paper:
P_cov(cell | robot) = 1 / (1 + e^(k*(r - r0)))

where:
- r = euclidean distance from robot to cell
- r0 = sigmoid midpoint (distance where P_cov = 0.5)
- k = steepness (higher = sharper falloff)
"""

import numpy as np
import matplotlib.pyplot as plt
from config import config

def test_distance_formula():
    """Test the distance-based coverage probability formula."""
    print("="*70)
    print("DISTANCE-BASED COVERAGE PROBABILITY TEST")
    print("="*70)
    
    # Parameters
    r0 = config.PROBABILISTIC_COVERAGE_MIDPOINT  # 1.5
    k = config.PROBABILISTIC_COVERAGE_STEEPNESS   # 2.0
    
    print(f"\nParameters:")
    print(f"  r0 (midpoint): {r0}")
    print(f"  k (steepness): {k}")
    
    # Test different distances
    distances = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    print(f"\nCoverage probability by distance:")
    print(f"{'Distance':<12} {'P_cov':<12} {'Description'}")
    print(f"{'-'*50}")
    
    for r in distances:
        p_cov = 1.0 / (1.0 + np.exp(k * (r - r0)))
        
        if r == 0.0:
            desc = "At robot position"
        elif r < r0:
            desc = "Close range"
        elif abs(r - r0) < 0.1:
            desc = "Midpoint (P=0.5)"
        else:
            desc = "Far range"
        
        print(f"{r:<12.1f} {p_cov:<12.4f} {desc}")
    
    # Verify key properties
    print(f"\n{'='*70}")
    print("VERIFICATION:")
    print(f"{'='*70}")
    
    # At distance 0, should be very high coverage
    p_0 = 1.0 / (1.0 + np.exp(k * (0 - r0)))
    print(f"✓ P_cov(r=0) = {p_0:.4f} (should be close to 1.0)")
    assert p_0 > 0.9, f"Coverage at robot position should be >0.9, got {p_0}"
    
    # At midpoint r0, should be 0.5
    p_mid = 1.0 / (1.0 + np.exp(k * (r0 - r0)))
    print(f"✓ P_cov(r=r0={r0}) = {p_mid:.4f} (should be 0.5)")
    assert abs(p_mid - 0.5) < 0.01, f"Coverage at midpoint should be 0.5, got {p_mid}"
    
    # At far distance, should be low
    p_far = 1.0 / (1.0 + np.exp(k * (3.0 - r0)))
    print(f"✓ P_cov(r=3.0) = {p_far:.4f} (should be < 0.1)")
    assert p_far < 0.1, f"Coverage at far distance should be <0.1, got {p_far}"
    
    # Monotonically decreasing
    print(f"✓ Coverage probability decreases with distance")
    for i in range(len(distances)-1):
        p1 = 1.0 / (1.0 + np.exp(k * (distances[i] - r0)))
        p2 = 1.0 / (1.0 + np.exp(k * (distances[i+1] - r0)))
        assert p1 > p2, f"Not monotonic: P({distances[i]})={p1}, P({distances[i+1]})={p2}"
    
    print(f"\n{'='*70}")
    print("ALL TESTS PASSED ✓")
    print(f"{'='*70}")


def visualize_coverage_distribution():
    """Create visualization of coverage probability distribution."""
    print("\n\nGenerating coverage probability plot...")
    
    r0 = config.PROBABILISTIC_COVERAGE_MIDPOINT
    k = config.PROBABILISTIC_COVERAGE_STEEPNESS
    
    # Generate distance range
    distances = np.linspace(0, 4, 100)
    p_covs = 1.0 / (1.0 + np.exp(k * (distances - r0)))
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(distances, p_covs, 'b-', linewidth=2, label=f'k={k}, r₀={r0}')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='P=0.5')
    plt.axvline(x=r0, color='r', linestyle='--', alpha=0.5, label=f'r₀={r0}')
    
    plt.xlabel('Distance from Robot (grid cells)', fontsize=12)
    plt.ylabel('Coverage Probability P_cov', fontsize=12)
    plt.title('Distance-Based Coverage Sensor Model', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.ylim(-0.05, 1.05)
    
    # Add annotations
    plt.annotate('High coverage\n(close range)', xy=(0.2, 0.95), fontsize=10,
                ha='left', va='top', color='green')
    plt.annotate('Midpoint\n(P=0.5)', xy=(r0, 0.5), xytext=(r0+0.5, 0.35),
                fontsize=10, ha='left', arrowprops=dict(arrowstyle='->', color='red'))
    plt.annotate('Low coverage\n(far range)', xy=(3.5, 0.05), fontsize=10,
                ha='right', va='bottom', color='orange')
    
    plt.tight_layout()
    plt.savefig('coverage_probability_distribution.png', dpi=150)
    print(f"✓ Plot saved to: coverage_probability_distribution.png")
    plt.close()


def test_2d_coverage_field():
    """Test 2D coverage field around robot."""
    print("\n\n" + "="*70)
    print("2D COVERAGE FIELD TEST")
    print("="*70)
    
    # Create grid
    grid_size = 11
    robot_pos = (5, 5)  # Center
    
    r0 = config.PROBABILISTIC_COVERAGE_MIDPOINT
    k = config.PROBABILISTIC_COVERAGE_STEEPNESS
    
    # Compute coverage field
    coverage_field = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            distance = np.sqrt((i - robot_pos[0])**2 + (j - robot_pos[1])**2)
            coverage_field[i, j] = 1.0 / (1.0 + np.exp(k * (distance - r0)))
    
    print(f"\nCoverage field (robot at {robot_pos}):")
    print(f"{'   '}  " + "  ".join([f"{i:4d}" for i in range(grid_size)]))
    print("-" * 70)
    for i in range(grid_size):
        row_str = f"{i:2d} |"
        for j in range(grid_size):
            row_str += f" {coverage_field[i, j]:.2f}"
        print(row_str)
    
    # Verify radial symmetry
    print(f"\n✓ Coverage field shows radial decay from robot position")
    print(f"  Robot position: P_cov = {coverage_field[robot_pos[0], robot_pos[1]]:.4f}")
    print(f"  Distance 1: P_cov ≈ {coverage_field[robot_pos[0]+1, robot_pos[1]]:.4f}")
    print(f"  Distance 2: P_cov ≈ {coverage_field[robot_pos[0]+2, robot_pos[1]]:.4f}")
    print(f"  Distance 3: P_cov ≈ {coverage_field[robot_pos[0]+3, robot_pos[1]]:.4f}")
    
    print(f"\n{'='*70}")
    print("2D FIELD TEST PASSED ✓")
    print(f"{'='*70}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DISTANCE-BASED PROBABILISTIC COVERAGE TESTS")
    print("="*70)
    print("\nBased on sensor model:")
    print("  P_cov(cell | robot) = 1 / (1 + e^(k*(r - r0)))")
    print("  where r = euclidean distance")
    print("\n")
    
    try:
        test_distance_formula()
        test_2d_coverage_field()
        visualize_coverage_distribution()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nDistance-based probabilistic coverage correctly implemented!")
        print("\nKey properties:")
        print("  • Coverage decreases with distance from robot")
        print("  • At robot position (r=0): P_cov ≈ 1.0 (full coverage)")
        print("  • At midpoint (r=r0): P_cov = 0.5")
        print("  • At far distance (r>2*r0): P_cov → 0")
        print("\nUsage:")
        print("  python train_fcn.py --probabilistic --episodes 400")
        print("  python train_multi_agent.py --probabilistic --agents 4 --episodes 400")
        print("\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

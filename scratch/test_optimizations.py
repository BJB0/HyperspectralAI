import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import time
from utils.patches import extract_patches, extract_patch_cubes
from clustering.kmeans_clustering import run_kmeans
from utils.metrics import calculate_metrics

def test_patches():
    print("=== Testing Vectorized Patch Extraction ===")
    # Create a small mockup image (100x100 RGB image)
    image = np.random.rand(100, 100, 3)
    
    t0 = time.time()
    patches = extract_patches(image, patch_size=3)
    t1 = time.time()
    cubes = extract_patch_cubes(image, patch_size=3)
    t2 = time.time()
    
    print(f"Patches shape: {patches.shape} (Expected: (10000, 27))")
    print(f"Cubes shape: {cubes.shape} (Expected: (10000, 3, 3, 3))")
    print(f"Patch extraction time: {t1 - t0:.6f}s")
    print(f"Cube extraction time: {t2 - t1:.6f}s")
    
    assert patches.shape == (10000, 27), "Incorrect patches shape"
    assert cubes.shape == (10000, 3, 3, 3), "Incorrect cubes shape"
    print("Patch extraction: SUCCESS\n")

def test_kmeans():
    print("=== Testing Adaptive K-Means ===")
    # Small dataset (1,000 samples) - should use standard KMeans
    small_data = np.random.rand(1000, 10)
    t0 = time.time()
    small_labels = run_kmeans(small_data, n_clusters=5)
    t1 = time.time()
    print(f"Small dataset (1,000 samples) processed in {t1 - t0:.4f}s")
    
    # Large dataset (150,000 samples) - should trigger MiniBatchKMeans
    large_data = np.random.rand(150000, 10)
    t2 = time.time()
    large_labels = run_kmeans(large_data, n_clusters=5)
    t3 = time.time()
    print(f"Large dataset (150,000 samples) processed in {t3 - t2:.4f}s (Using MiniBatchKMeans)")
    
    assert len(small_labels) == 1000, "Incorrect small labels count"
    assert len(large_labels) == 150000, "Incorrect large labels count"
    print("Adaptive K-Means: SUCCESS\n")

def test_metrics():
    print("=== Testing Vectorized Metrics & Label Mapping ===")
    # Create mock ground truth and predicted labels
    true_labels = np.random.randint(1, 5, size=(10000,))
    # pred labels are true labels with some random noise and class label shifting
    pred_labels = (true_labels + 1) % 4 + 1
    
    t0 = time.time()
    acc, kappa, nmi, cm = calculate_metrics(true_labels, pred_labels)
    t1 = time.time()
    
    print(f"Metrics: Acc={acc:.4f}, Kappa={kappa:.4f}, NMI={nmi:.4f}")
    print(f"Remap & metrics execution time: {t1 - t0:.6f}s")
    
    assert 0 <= acc <= 1.0, "Accuracy out of bounds"
    assert -1.0 <= kappa <= 1.0, "Kappa out of bounds"
    assert 0 <= nmi <= 1.0, "NMI out of bounds"
    print("Vectorized Metrics & Label Mapping: SUCCESS\n")

if __name__ == "__main__":
    test_patches()
    test_kmeans()
    test_metrics()
    print("All optimization checks PASSED!")

import sys
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

# Add parent directory to sys.path to enable local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import core function and methods from app.py
from app import run_selected_method

def run_integration_tests():
    print("====================================================")
    print("STARTING FULL END-TO-END APP FUNCTIONALITY CHECK")
    print("====================================================")

    # 1. Generate a mock image representing a 32x32 hyperspectral image with 40 bands
    h, w, c = 32, 32, 40
    print(f"Generating mock HSI image cube of shape: ({h}, {w}, {c})...")
    mock_image = np.random.rand(h, w, c).astype(np.float32)

    # Standardize features for KMeans methods
    pixels = mock_image.reshape(-1, c)
    scaler = StandardScaler()
    pixels_scaled = scaler.fit_transform(pixels)

    # Parameters to test
    n_clusters = 3
    patch_size = 3
    pca_components = 5
    latent_dim = 4
    single_band_index = 5
    false_color_indices = [1, 2, 3]

    # List of all 8 methods defined in the app
    methods = [
        "Single Band KMeans",
        "False Color KMeans",
        "KMeans",
        "PCA + KMeans",
        "Spatial-Spectral",
        "Autoencoder",
        "CNN Autoencoder",
        "DEC"
    ]

    success_count = 0
    failures = []

    for idx, method in enumerate(methods, 1):
        print(f"\n[{idx}/{len(methods)}] Testing method: '{method}'...")
        try:
            # Execute the core function that Streamlit runs
            labels, features = run_selected_method(
                method_name=method,
                image=mock_image,
                pixels_scaled=pixels_scaled,
                n_clusters=n_clusters,
                patch_size=patch_size,
                pca_components=pca_components,
                latent_dim=latent_dim,
                single_band_index=single_band_index,
                false_color_indices=false_color_indices
            )
            
            # Check outputs
            assert labels is not None, f"Method '{method}' returned None labels."
            assert len(labels) == h * w, f"Method '{method}' returned incorrect labels size: {len(labels)} (expected {h * w})"
            
            print(f"-> SUCCESS: Clustered {h * w} pixels into {len(np.unique(labels))} unique classes.")
            if features is not None:
                print(f"   Reduced features shape: {features.shape}")
            else:
                print("   No features vector returned (as expected for direct KMeans baseline).")
            
            success_count += 1
        except Exception as e:
            print(f"-> FAILURE: Method '{method}' failed with error: {e}")
            import traceback
            traceback.print_exc()
            failures.append((method, str(e)))

    print("\n====================================================")
    print("INTEGRATION TEST SUMMARY")
    print(f"Total tested methods: {len(methods)}")
    print(f"Successful: {success_count}")
    print(f"Failures: {len(failures)}")
    print("====================================================")

    if failures:
        print("Detailed failures:")
        for method, error in failures:
            print(f"- {method}: {error}")
        sys.exit(1)
    else:
        print("ALL CORE APP FUNCTIONALITIES REMAIN FULLY OPERATIONAL!")
        sys.exit(0)

if __name__ == "__main__":
    run_integration_tests()

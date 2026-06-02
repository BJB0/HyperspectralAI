import numpy as np

def extract_patches(image, patch_size=3):

    pad = patch_size // 2

    h, w, c = image.shape

    padded = np.pad(
        image,
        ((pad, pad), (pad, pad), (0, 0)),
        mode='reflect'
    )

    try:
        from numpy.lib.stride_tricks import sliding_window_view
        # Extract sliding windows: shape (h, w, 1, patch_size, patch_size, c)
        windows = sliding_window_view(padded, (patch_size, patch_size, c))
        # Reshape to (h * w, patch_size * patch_size * c)
        return windows.reshape(h * w, -1)
    except (ImportError, AttributeError):
        # Fallback to loop if numpy doesn't support sliding_window_view
        patches = []

        for i in range(h):

            for j in range(w):

                patch = padded[
                    i:i+patch_size,
                    j:j+patch_size
                ]

                patches.append(
                    patch.flatten()
                )

        return np.array(patches)


def extract_patch_cubes(image, patch_size=3):

    pad = patch_size // 2

    h, w, c = image.shape

    padded = np.pad(
        image,
        ((pad, pad), (pad, pad), (0, 0)),
        mode='reflect'
    )

    try:
        from numpy.lib.stride_tricks import sliding_window_view
        # Extract sliding windows: shape (h, w, 1, patch_size, patch_size, c)
        windows = sliding_window_view(padded, (patch_size, patch_size, c))
        # Reshape to (h * w, patch_size, patch_size, c)
        return windows.reshape(h * w, patch_size, patch_size, c)
    except (ImportError, AttributeError):
        # Fallback to loop if numpy doesn't support sliding_window_view
        patches = []

        for i in range(h):

            for j in range(w):

                patch = padded[
                    i:i+patch_size,
                    j:j+patch_size,
                    :
                ]

                patches.append(patch)

        return np.array(patches)

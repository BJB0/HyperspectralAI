import numpy as np

def extract_patches(image, patch_size=3):

    pad = patch_size // 2

    h, w, c = image.shape

    padded = np.pad(
        image,
        ((pad, pad), (pad, pad), (0, 0)),
        mode='reflect'
    )

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
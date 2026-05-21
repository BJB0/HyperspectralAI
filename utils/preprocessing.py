import numpy as np

def detect_image_type(data):

    if len(data.shape) == 3:

        if data.shape[2] <= 4:
            return "RGB"

        else:
            return "HSI"

    return "UNKNOWN"
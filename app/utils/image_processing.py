import cv2
import numpy as np

def apply_clahe(image: np.ndarray, clip_limit: float = 4.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:

    if image is None:
        return None
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

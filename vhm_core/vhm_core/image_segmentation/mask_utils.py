from pathlib import Path
import cv2
import numpy as np


def save_mask(mask: np.ndarray, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), mask)
    return str(output_path)

def apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def save_crop(image: np.ndarray, mask: np.ndarray, bbox: list[int], output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h))

    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"Invalid crop bbox: {bbox}")
    
    masked_image = apply_mask_to_image(image, mask)

    crop = masked_image[y1:y2, x1:x2]

    if crop.size == 0:
        raise RuntimeError(f"Empty crop from bbox: {bbox}")

    cv2.imwrite(str(output_path), crop)
    return str(output_path)
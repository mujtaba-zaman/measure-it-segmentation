"""Image transforms shared by calibration, preprocessing and inference."""
import cv2


def normalize_landscape(image):
    """Rotate portrait frames clockwise so all frames share landscape axes."""
    height, width = image.shape[:2]
    if height > width:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image

import cv2
def upscale_image_for_better_ocr(img, kernel_size=(3, 3), sigmaX=0, save_debug=False):
    """
    Dynamically upscale image to improve OCR accuracy with controllable blurring.
    
    Args:
        img: Input image
        kernel_size (tuple): Size of the Gaussian kernel (e.g., (3, 3))
        sigmaX (float): Standard deviation in X direction; if 0, computed from kernel size
        save_debug (bool): Save intermediate blurred image for inspection
    
    Returns:
        Sharpened upscaled image
    """
    height, width = img.shape[:2]

    # Dynamic scaling based on bubble size
    if min(width, height) < 120:
        scale_factor = 3.0   # Tiny bubble → strong upscale
    elif max(width, height) > 200:
        scale_factor = 1.5   # Big bubble → light upscale
    else:
        scale_factor = 2.0   # Default

    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Upscale with high-quality interpolation
    upscaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Apply Gaussian blur with controllable parameters
    blurred = cv2.GaussianBlur(upscaled, kernel_size, sigmaX)

    # Sharpen the image
    sharpened = cv2.addWeighted(blurred, 1.5, cv2.GaussianBlur(blurred, kernel_size, sigmaX), -0.5, 0)

    if save_debug:
        cv2.imwrite("debug_blurred.png", blurred)

    return sharpened
import numpy as np

def resize_for_yolo(img, target_size=(640, 640), keep_aspect_ratio=True):
    h, w = img.shape[:2]
    if keep_aspect_ratio:
        scale = min(target_size[0] / w, target_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((target_size[1], target_size[0]), dtype=img.dtype)
        x_offset = (target_size[0] - new_w) // 2
        y_offset = (target_size[1] - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        return canvas
    else:
        return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

def preprocess_image_for_ocr(img, upscale=True, scale_factor=2.0, kernel_size=(3, 3), sigmaX=0, save_debug=False, for_yolo=False, grayscale=True):
    """
    Enhanced preprocessing for OCR with optional grayscale, upscaling, and YOLO resizing.

    Args:
        img: Input image
        upscale (bool): Whether to upscale the image
        scale_factor (float): Factor to scale the image if upscaling
        kernel_size (tuple): Size of the Gaussian kernel for blurring
        sigmaX (float): Standard deviation for Gaussian blur
        save_debug (bool): Save intermediate preprocessed image
        for_yolo (bool): Resize image to YOLO-compatible dimensions
        grayscale (bool): Convert image to grayscale

    Returns:
        Preprocessed image
    """
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if upscale and (img.shape[0] < 100 or img.shape[1] < 100):
        img = upscale_image_for_better_ocr(img, kernel_size, sigmaX, save_debug)

    if grayscale:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        img = cv2.bilateralFilter(img, 9, 75, 75)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    if for_yolo:
        img = resize_for_yolo(img, target_size=(640, 640))

    if save_debug:
        cv2.imwrite("debug_preprocessed.png", img)

    return img

import os
import cv2


def save_bubble_images(image, boxes, output_dir="bubble_crops"):
    """
    Save cropped bubble images for debugging.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        bubble_crop = image[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(output_dir, f"bubble_{i+1}.png"), bubble_crop)

def _sort_bubbles_by_reading_order(bubble_coords, alpha=0.5):
    """
    Sort bubbles using a weighted reading order heuristic.
    """
    if not bubble_coords:
        return []

    scores = [(i, y + alpha * x) for i, (y, x) in enumerate(bubble_coords)]
    scores.sort(key=lambda item: item[1])
    return [i for i, _ in scores]
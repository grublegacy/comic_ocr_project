from ultralytics import YOLO
import easyocr
from preprocessing import preprocess_image_for_ocr
from utils import save_bubble_images, _sort_bubbles_by_reading_order
from config import MODEL_PATH  
import cv2

def extract_text_from_comic_page(image_path, model_path = MODEL_PATH,
                                save_crops=False, upscale=True, scale_factor=2.0, kernel_size=(3, 3), sigmaX=0, save_debug=False):
    #Load model
    model = YOLO(model_path)
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")

    # Perform detection
    results = model(image)

    extracted_texts = []
    bubble_coords = []
    reader = easyocr.Reader(['en'])
    
    # Process each detection
    for result in results:
        boxes = result.boxes
        if save_crops:
            save_bubble_images(image, boxes)

        for box in boxes:
            if int(box.cls) == 0:  # Assuming class 0 is speech bubble
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bubble_crop = image[y1:y2, x1:x2]
                
                # Preprocess with upscaling and controllable blur
                preprocessed = preprocess_image_for_ocr(bubble_crop, upscale=upscale, scale_factor=scale_factor,
                                                      kernel_size=kernel_size, sigmaX=sigmaX, save_debug=save_debug)
                
                # Extract text with EasyOCR
                try:
                    text_results = reader.readtext(preprocessed, detail=0)
                except Exception as e:
                    print(f"OCR failed on bubble at ({x1}, {y1}, {x2}, {y2}): {e}")
                    continue

                text = " ".join(text_results) if isinstance(text_results, list) else text_results

                # Clean the OCR text (using spellcheck from your module)
                cleaned_text = text  # Placeholder; replace with spellcheck.spellcheck_sentences if defined
                if cleaned_text:
                    extracted_texts.append(cleaned_text)
                    bubble_coords.append((y1, x1))  # Store top-left coordinates for ordering

    # Sort by reading order
    if bubble_coords:
        sorted_indices = _sort_bubbles_by_reading_order(bubble_coords)
        extracted_texts = [extracted_texts[i].lower() for i in sorted_indices]

    return extracted_texts
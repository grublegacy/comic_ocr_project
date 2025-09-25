from ultralytics import YOLO
import easyocr
import cv2
import re
import spacy
import language_tool_python
from spellchecker import SpellChecker
from preprocessing import preprocess_image_for_ocr
from utils import save_bubble_images, _sort_bubbles_by_reading_order
from config import MODEL_PATH

# Initialize models
spell = SpellChecker()
reader = easyocr.Reader(['en'])

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Spellcheck helpers
def apply_spellchecker(text):
    words = text.split()
    corrected = [spell.correction(word) or word for word in words]
    return " ".join(corrected)

def detect_names(text):
    doc = nlp(text)
    return {ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG"}}

def preserve_entities(text, entities):
    placeholders = {name: f"__ENT{i}__" for i, name in enumerate(sorted(entities, key=len, reverse=True))}
    for name, placeholder in placeholders.items():
        pattern = rf"\b{re.escape(name)}\b"
        text = re.sub(pattern, placeholder, text)
    return text, placeholders

def restore_entities(text, placeholders):
    for name, placeholder in placeholders.items():
        text = text.replace(placeholder, name)
    return text

def spellcheck_sentences(sentences):
    corrected = []
    with language_tool_python.LanguageTool("en-US") as tool:
        for sentence in sentences:
            if not re.search(r'[a-zA-Z]', sentence):
                corrected.append(sentence)
                continue
            names = detect_names(sentence)
            temp_sentence, placeholders = preserve_entities(sentence, names)

            matches = tool.check(temp_sentence)
            corrected_temp = language_tool_python.utils.correct(temp_sentence, matches)
            corrected_temp = apply_spellchecker(corrected_temp)
            final_sentence = restore_entities(corrected_temp, placeholders)

            corrected.append(final_sentence)
    return corrected

# Main OCR function
def extract_text_from_comic_page(image_path, model_path=MODEL_PATH,
                                save_crops=False, upscale=True, scale_factor=2.0,
                                kernel_size=(3, 3), sigmaX=0, save_debug=False):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")

    model = YOLO(model_path)
    results = model(image)

    raw_texts = []
    bubble_coords = []

    for result in results:
        boxes = result.boxes
        if save_crops:
            save_bubble_images(image, boxes)

        for box in boxes:
            if int(box.cls) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bubble_crop = image[y1:y2, x1:x2]

                preprocessed = preprocess_image_for_ocr(
                    bubble_crop, upscale=upscale, scale_factor=scale_factor,
                    kernel_size=kernel_size, sigmaX=sigmaX, save_debug=save_debug
                )

                try:
                    text_results = reader.readtext(preprocessed, detail=0)
                except Exception as e:
                    print(f"OCR failed at ({x1},{y1}): {e}")
                    continue

                text = " ".join(text_results) if isinstance(text_results, list) else text_results
                if text:
                    raw_texts.append(text)
                    bubble_coords.append((y1, x1))

    if bubble_coords:
        sorted_indices = _sort_bubbles_by_reading_order(bubble_coords)
        sorted_texts = [raw_texts[i] for i in sorted_indices]
        cleaned_texts = spellcheck_sentences(sorted_texts)
        return [t.lower() for t in cleaned_texts]

    return []

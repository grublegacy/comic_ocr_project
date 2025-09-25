from detection import extract_text_from_comic_page
from spellcheck import spellcheck_sentences
from config import MODEL_PATH, IMAGE_PATHS
import traceback

if __name__ == "__main__":
    total_bubbles = 0
    total_words = 0
    
    try:
        for comic_image in IMAGE_PATHS:
            print(f"\nProcessing: {comic_image}")
            texts = extract_text_from_comic_page(comic_image, model_path=MODEL_PATH, save_crops=True,
                                                 upscale=True, scale_factor=2.0, kernel_size=(5, 5), sigmaX=1.0,
                                                 save_debug=True)
            corrected_texts = spellcheck_sentences(texts)
            
            print(f" Extracted {len(corrected_texts)} text bubbles:")
            print("-" * 30)
            for i, text in enumerate(corrected_texts, 1):
                print(f"Bubble {i}: {text}")
            
            total_bubbles += len(corrected_texts)
            total_words += sum(len(text.split()) for text in corrected_texts)
        
        print(f"\n SUMMARY:")
        print(f"Total bubbles: {total_bubbles}")
        print(f"Total words: {total_words}")
        print(f"Average words per bubble: {total_words / total_bubbles if total_bubbles > 0 else 0:.1f}")


    
    except Exception as e:
        print(f" Error: {e}")
        traceback.print_exc()
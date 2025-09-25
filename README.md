
# Comic OCR Project

A Python-based tool for extracting and correcting text from comic book speech bubbles using YOLO, EasyOCR, and spellchecking.

---

## Features

- Detects speech bubbles in comic pages .
- Extracts text using .
- Automatically corrects spelling and grammar with **LanguageTool** and **PySpellChecker**.
- Handles small/pixelated bubbles with **dynamic upscaling** and preprocessing.
- Preserves named entities (characters' names, organizations) during spellchecking.
- Provides a **web interface** using **Flask** for uploading and viewing results.

---

## Demo
1.Clone the repository:
-git clone https://github.com/username/comic_ocr_project.git
-cd comic_ocr_project

2.download yolo model and images
-Model file: comic-speech-bubble-detector.pt
-Place it in data/models/
-googledrive  link here: https://drive.google.com/drive/folders/1MX7svdwIQtgihNN53cYcYFcG2FQ4zAYl?usp=drive_link

3. install depndencies:
-pip install -r requirements.txt
-#download en_core_web_sm model 
-python -m spacy download en_core_web_sm

4. run src/comic_ocr/app.py
-upload test images from data/images



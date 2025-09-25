import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'data', 'models', 'comic-speech-bubble-detector.pt')
IMAGE_PATHS = [
    os.path.join(BASE_DIR, '..', 'data', 'images', 'comics2.jpg'),
    os.path.join(BASE_DIR, '..', 'data', 'images', 'kawaiiju.webp'),
    os.path.join(BASE_DIR, '..', 'data', 'images', 'paranorthern-page-10.png')
]

print(IMAGE_PATHS)
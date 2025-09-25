
from flask import Flask, render_template, request, redirect
import os
from werkzeug.utils import secure_filename
from detection import extract_text_from_comic_page
from spellcheck import spellcheck_sentences
from config import MODEL_PATH

app = Flask(__name__, template_folder='../../templates', static_folder='../../static')

# Configure upload settings
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files' not in request.files:
        return redirect(request.url)
    
    files = request.files.getlist('files')
    results = []
    total_bubbles = 0
    total_words = 0
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process the image
            texts = extract_text_from_comic_page(
                file_path,
                model_path=MODEL_PATH,
                save_crops=True,
                upscale=True,
                scale_factor=2.0,
                kernel_size=(5, 5),
                sigmaX=1.0,
                save_debug=True
            )
            corrected_texts = spellcheck_sentences(texts)
            
            # Collect results
            results.append({
                'image': filename,
                'texts': corrected_texts,
                'bubble_count': len(corrected_texts),
                'word_count': sum(len(text.split()) for text in corrected_texts)
            })
            
            total_bubbles += len(corrected_texts)
            total_words += sum(len(text.split()) for text in corrected_texts)
    
    # Calculate summary
    summary = {
        'total_bubbles': total_bubbles,
        'total_words': total_words,
        'avg_words_per_bubble': round(total_words / total_bubbles, 1) if total_bubbles > 0 else 0
    }
    
    return render_template('results.html', results=results, summary=summary)

if __name__ == '__main__':
    app.run()
from flask import Flask, request, redirect, url_for, render_template
from tensorflow.keras.preprocessing import image as kp_image
import os
from werkzeug.utils import secure_filename
import style_transfer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        content_file = request.files['content']
        style_file = request.files['style']
        content_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(content_file.filename))
        style_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(style_file.filename))
        content_file.save(content_path)
        style_file.save(style_path)

        output_path = 'static/output.png'

        # Perform style transfer
        generated_image = style_transfer.train_style_transfer(content_path, style_path)

        # Save the generated image
        kp_image.save_img(output_path, generated_image)

        return render_template('result.html', output_image=output_path)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
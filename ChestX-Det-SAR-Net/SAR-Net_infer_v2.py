from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2
import os
from utils import *
from flask import send_from_directory
RESULTS_FOLDER = '/root/hackathon2022/OraichainHackathon2022/ChestX-Det-SAR-Net/results'
# Initialize the Flask application
app = Flask(__name__)

@app.route('/download')
def download():
    filename = request.args.get('filename')
    return send_from_directory(RESULTS_FOLDER, filename)

# route http posts to this method
@app.route('/process', methods=['POST'])
def process():
    hash_code = request.form['image']
    filename = hash_code
    # # convert string of image data to uint8
    # # decode image
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    # cv2.imwrite(file.filename, img)
    image_shape = img.shape[:2]
    # DO PREDICT
    response = do_predict(filename)
    img_drawed = visualize(img, response)
    cv2.imwrite(os.path.join(RESULTS_FOLDER, filename), img_drawed)
    return filename

@app.route('/upload_file', methods=['POST'])
def upload_file():
    file = request.files['image']
    file.save(os.path.join(RESULTS_FOLDER, filename))
    return file.filename
    
if __name__ == "__main__":
    # start flask app
    app.run(host="0.0.0.0", port=5000)

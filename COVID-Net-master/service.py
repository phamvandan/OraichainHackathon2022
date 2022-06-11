from flask import Flask, request, Response
import numpy as np
import cv2
import os
import pandas as pd
import pickle
import jsonpickle
from utils import do_predict

# Initialize the Flask application
app = Flask(__name__)
# route http posts to this method
@app.route('/process', methods=['POST'])
def process():
    hash_code = request.form['image']
    filename = hash_code
    response = do_predict(filename)
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route('/upload_file', methods=['POST'])
def upload_file():
    file = request.files['image']
    file.save(file.filename)
    return file.filename

if __name__ == "__main__":
    # start flask app
    app.run(host="0.0.0.0", port=5010)

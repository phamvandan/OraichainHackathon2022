from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2
import os
from utils import *
from flask import send_from_directory
RESULTS_FOLDER = 'results'
# Initialize the Flask application
app = Flask(__name__)

@app.route('/download')
def download():
    filename = request.args.get('filename')
    return send_from_directory(RESULTS_FOLDER, filename)

# route http posts to this method
@app.route('/process', methods=['POST'])
def process():
    file = request.files['image']
    # # convert string of image data to uint8
    nparr = np.fromstring(file.read(), np.uint8)
    # # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # cv2.imwrite(file.filename, img)
    image_shape = img.shape[:2]
    # DO PREDICT
    response = do_predict(file.filename)
    img_drawed = visualize(img, response)
    cv2.imwrite(os.path.join(RESULTS_FOLDER, file.filename), img_drawed)
    response['image_download_link'] = '/download?filename=' + file.filename
    response['image_shape'] = list(image_shape)
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")

if __name__ == "__main__":
    # start flask app
    app.run(host="0.0.0.0", port=5000)
from flask import Flask, request, Response, jsonify
import numpy as np
import cv2
import model
from model import *


app = Flask(__name__)

@app.route('/prediction', methods=['POST'])
def prediction():
    """
        Get request, make a prediction and answer back to sender
    """

    # get data
    r = request
    img_url = r.data

    # predict
    prediction = model.predict(img_url)

    # build a response dict to send back to client
    response = { 'message': prediction }

    return jsonify(str(response))


if __name__ == '__main__':
    model = Model("train_cnn.h5")
    app.run(host='0.0.0.0', port=80)
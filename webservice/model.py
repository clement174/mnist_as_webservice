import keras
import numpy as np
from keras.models import load_model
import cv2
import urllib
from urllib.request import Request, urlopen


class Model(object):

    def __init__(self, model_file):
        """
            Set image_size and load model
            call make_predict_function() on model to avoid a bug
        """

        self.image_size = 28
        self.model = load_model(model_file)
        self.model._make_predict_function()


    def preprocess_data(self, img_url):
        """
            Decode and load image from url.
            Preprocess accordingly to our model input shape
        """

        # get image
        req = Request(img_url.decode('ASCII'), headers={'User-Agent': 'Mozilla/5.0'})
        open_req = urlopen(req)
        #decode image
        arr = np.asarray(bytearray(open_req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr , cv2.IMREAD_GRAYSCALE)
        # resize
        img = cv2.resize(img, (self.image_size, self.image_size))
        # normalize
        X = img.astype('float32')
        X = X / 255
        X = X.reshape(1, self.image_size, self.image_size, 1)

        return X


    def predict(self, img_url):
        """
            Main function, take url of an image (digit in this case) and return prediction (from 0 to 9)
        """

        X = self.preprocess_data(img_url)
        prediction = self.model.predict(X)
        label = np.argmax(prediction, axis=1)[0]

        return label
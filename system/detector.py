import cv2
import numpy as np
from tensorflow.keras.models import load_model

class PopupDetector:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.img_size = (224, 224)
        
    def classify(self, img):
        img = cv2.resize(img, self.img_size)
        img_array = np.expand_dims(img, axis=0) / 255.0
        prediction = self.model.predict(img_array)[0][0]
        return "fake" if prediction > 0.7 else "real", float(prediction)
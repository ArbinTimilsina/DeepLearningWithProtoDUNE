import os
import configparser
import numpy as np
from keras.callbacks import Callback
from tools.data_tools import get_data_generator, preprocess_feature, preprocess_label

config = configparser.ConfigParser()
config_path = os.path.join("configurations", "master_configuration.ini")
config.read(config_path)

IMAGE_WIDTH = int(config["DEFAULT"]["IMAGE_WIDTH"])
IMAGE_HEIGHT = int(config["DEFAULT"]["IMAGE_HEIGHT"])
IMAGE_DEPTH = int(config["DEFAULT"]["IMAGE_DEPTH"])
CLASS_NAMES = config["DEFAULT"]["CLASS_NAMES"].split()
FEATURE_FILE_TESTING = config["DEFAULT"]["FEATURE_FILE_TESTING"]
LABEL_FILE_TESTING = config["DEFAULT"]["LABEL_FILE_TESTING"]

generator_testing = get_data_generator(FEATURE_FILE_TESTING, LABEL_FILE_TESTING)
X, y = next(generator_testing)
X_preprocessed = preprocess_feature(X, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH)
y_preprocessed = preprocess_label(y, IMAGE_WIDTH, IMAGE_HEIGHT, len(CLASS_NAMES))
y_preprocessed_max = np.argmax(y_preprocessed, axis=3)

class PredictionHistory(Callback):
    def __init__(self, model):
        self.feature_image = []
        self.label_image = []
        self.prediction_image = []
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        prediction = self.model.predict_on_batch(X_preprocessed)
        prediction_max = np.argmax(prediction, axis=3)

        self.feature_image.append(X_preprocessed.reshape(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH))
        self.label_image.append(y_preprocessed_max.reshape(IMAGE_WIDTH, IMAGE_HEIGHT))
        self.prediction_image.append(prediction_max.reshape(IMAGE_WIDTH, IMAGE_HEIGHT))

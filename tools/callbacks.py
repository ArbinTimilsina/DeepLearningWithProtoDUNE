import os
import numpy as np
import configparser
from keras import backend as K
from keras.callbacks import Callback
from tools.data_tools import get_data_generator, preprocess_feature, preprocess_label


class PredictionsCallback(Callback):
    def __init__(self, model):
        self.feature_image = []
        self.label_image = []
        self.prediction_image = []
        self.model = model
        self.init_predictions()
        
    def init_predictions(self):
        config = configparser.ConfigParser()
        config_path = os.path.join("configurations", "master_configuration.ini")
        config.read(config_path)

        self.IMAGE_WIDTH = int(config["DEFAULT"]["IMAGE_WIDTH"])
        self.IMAGE_HEIGHT = int(config["DEFAULT"]["IMAGE_HEIGHT"])
        self.IMAGE_DEPTH = int(config["DEFAULT"]["IMAGE_DEPTH"])
        CLASS_NAMES = config["DEFAULT"]["CLASS_NAMES"].split()
        FEATURE_FILE_TESTING = config["DEFAULT"]["FEATURE_FILE_TESTING"]
        LABEL_FILE_TESTING = config["DEFAULT"]["LABEL_FILE_TESTING"]

        generator_testing = get_data_generator(FEATURE_FILE_TESTING, LABEL_FILE_TESTING)
        X, y = next(generator_testing)
        y_preprocessed = preprocess_label(y, self.IMAGE_WIDTH, self.IMAGE_HEIGHT, len(CLASS_NAMES))
        self.y_preprocessed_max = np.argmax(y_preprocessed, axis=-1)
        self.X_preprocessed = preprocess_feature(X, self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.IMAGE_DEPTH)
    
    def on_epoch_end(self, epoch, logs=None):
        prediction = self.model.predict_on_batch(self.X_preprocessed)
        prediction_max = np.argmax(prediction, axis=-1)

        self.feature_image.append(self.X_preprocessed.reshape(self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.IMAGE_DEPTH))
        self.label_image.append(self.y_preprocessed_max.reshape(self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        self.prediction_image.append(prediction_max.reshape(self.IMAGE_WIDTH, self.IMAGE_HEIGHT))

class WeightsCallback(Callback):
    def __init__(self, weights, max_epoch, max_weight):
        self.weights = weights
        self.max_epoch = max_epoch
        self.max_weight = max_weight

    def on_epoch_end(self, epoch, logs=None):
        growth_rate = 1/(self.max_epoch**0.6)

        weight_0 = K.get_value(self.weights[0])

        weight_1 = K.get_value(self.weights[1])
        weight_1 = weight_1 + growth_rate * weight_1 * (1 - weight_1/self.max_weight)

        weight_2 = K.get_value(self.weights[2])
        weight_2 = weight_2 + growth_rate * weight_2 * (1 - weight_2/self.max_weight)

        K.set_value(self.weights, [weight_0, weight_1, weight_2])

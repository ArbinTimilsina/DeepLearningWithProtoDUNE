import os
import configparser
from keras.callbacks import Callback
from tools.data_tools import get_data_generator, preprocess_feature, preprocess_label

def init_predictions():
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
    y_preprocessed_max = K.argmax(y_preprocessed, axis=-1)

class PredictionsCallback(Callback):
    def __init__(self, model):
        self.feature_image = []
        self.label_image = []
        self.prediction_image = []
        self.model = model
        init_predictions()

    def on_epoch_end(self, epoch, logs=None):
        prediction = self.model.predict_on_batch(X_preprocessed)
        prediction_max = K.argmax(prediction, axis=-1)

        self.feature_image.append(X_preprocessed.reshape(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH))
        self.label_image.append(y_preprocessed_max.reshape(IMAGE_WIDTH, IMAGE_HEIGHT))
        self.prediction_image.append(prediction_max.reshape(IMAGE_WIDTH, IMAGE_HEIGHT))

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

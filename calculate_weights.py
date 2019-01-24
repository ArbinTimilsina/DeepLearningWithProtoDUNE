import os
import configparser
from tqdm import tqdm
from collections import Counter
from tools.data_tools import get_data_generator
from tools.plotting_tools import plot_weights_median

import numpy as np
from sklearn.utils import class_weight

def get_class_weights(y):
    """
    Returns the weights for each class based on the frequencies of the samplesself.
    For 3 classes with classA:10%, classB:50% and classC:40%, weights will be: {0: 1, 1: 5, 2: 4}
    The loss will be 5x more for miss-classifying classA than for classB and so on...
    """
    counter = Counter(y)

    minority = min(counter.values())
    return {cls: float(count) / float(minority) for cls, count in counter.items()}

def main():
    config = configparser.ConfigParser()
    config_path = os.path.join("configurations", "master_configuration.ini")
    config.read(config_path)
    print("\nReading info from configuration:")

    FEATURE_FILE_TRAINING = config["DEFAULT"]["FEATURE_FILE_TRAINING"]
    LABEL_FILE_TRAINING = config["DEFAULT"]["LABEL_FILE_TRAINING"]
    CLASS_NAMES = config["DEFAULT"]["CLASS_NAMES"].split()

    print("FEATURE_FILE_TRAINING: {}".format(FEATURE_FILE_TRAINING))
    print("LABEL_FILE_TRAINING: {}".format(LABEL_FILE_TRAINING))
    print("CLASS_NAMES: {}".format(CLASS_NAMES))
    print()

    iter_data = get_data_generator(FEATURE_FILE_TRAINING, LABEL_FILE_TRAINING)
    weights = [[],[],[]]
    for X, y in tqdm(iter_data):
        # Class weights will be given by n_samples / (n_classes * np.bincount(y))
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)
        for index, weight in enumerate(class_weights):
            weights[index].append(weight)

    ranges = [(0.34,0.5), (30, 150), (0, 14)]
    plot_path = os.path.join("plots", "extra", "weights_median.pdf")
    plot_weights_median(weights, ranges, CLASS_NAMES, plot_path)
    print("\nDone! Plot with median weights for each class is saved at {}!\n".format(plot_path))

if __name__ == "__main__":
    main()

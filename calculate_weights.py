import configparser
from collections import Counter
from tqdm import tqdm
from os import path
from get_data import get_data_generator
from plotting_tools import plot_weights_median

def get_class_weights(y):
    """
    Returns the weights for each class based on the frequencies of the samples
    For 3 classes with classA:10%, classB:50% and classC:40%, weights will be: {0:5, 1:1, 2:1.25}
    The loss will be 5x more for miss-classifying classA than for classB and so on...
    """
    counter = Counter(y)

    majority = max(counter.values())
    return {cls: float(majority) / count for cls, count in counter.items()}

def main():
    config = configparser.ConfigParser()
    config.read('configuration.ini')
    print("\nReading info from configuration")
    FEATURE_FILE_TRAINING = config["DEFAULT"]["FEATURE_FILE_TRAINING"]
    LABEL_FILE_TRAINING = config["DEFAULT"]["LABEL_FILE_TRAINING"]
    CLASSE_NAMES = config["DEFAULT"]["CLASSE_NAMES"].split()

    print("FEATURE_FILE_TRAINING: {}".format(FEATURE_FILE_TRAINING))
    print("LABEL_FILE_TRAINING: {}\n".format(LABEL_FILE_TRAINING))
    print("CLASSE_NAMES: {}\n".format(CLASSE_NAMES))

    iter_data = get_data_generator(FEATURE_FILE_TRAINING, LABEL_FILE_TRAINING)
    weights = [[],[],[]]
    for X, y in tqdm(iter_data):
        class_weights = get_class_weights(y)
        for index, weight in class_weights.items():
            weights[index].append(weight)

    ranges = [(0,2), (0,50), (0, 3500)]
    plot_path = path.join("plots", "weights_median.pdf")
    plot_weights_median(weights, ranges, CLASSE_NAMES, plot_path)
    print("\nDone! Plot with median weights for each class is saved at {}!\n".format(plot_path))

if __name__ == "__main__":
    main()

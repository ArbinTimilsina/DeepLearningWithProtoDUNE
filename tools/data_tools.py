import os
import csv
import numpy as np
from keras.utils import np_utils, Sequence

def get_data_generator(feature_file, label_file):
    """
    Allows to iterate over csv files.
    Generates one row at a time.
    """
    with open(os.path.join(feature_file, "feature.csv"), "r") as csv1, open(os.path.join(label_file, "label.csv"), "r") as csv2:
        reader1 = csv.reader(csv1)
        reader2 = csv.reader(csv2)
        # Skip the header row
        next(reader1)
        next(reader2)
        for row1, row2 in zip(reader1, reader2):
            array_row1 = np.array(row1, dtype=np.float)
            array_row2 = np.array(row2, dtype=np.int)
            yield array_row1, array_row2

def preprocess_feature(x, image_width, image_height, image_depth, normalize=False):
    """
    Feature is the adc values; scale it such that each value is between 0 and 1.
    """
    x_max = np.max(x)
    x = x/x_max

    if normalize:
        mean = np.mean(x)
        std = np.std(x)
        x -= mean
        x /= std

    return x.reshape(1, image_width, image_height, image_depth)

def preprocess_label(y, image_width, image_height, num_classes):
    return np_utils.to_categorical(y, num_classes=num_classes).reshape(1, image_width, image_height, num_classes)

class DataSequence(Sequence):
    """
    Although sequence are a safer way to do multiprocessing,
    use_multiprocessing=True in fit_generator is currently not supported here.
    """
    def __init__(self, feature_file, label_file,
                 image_width, image_height, image_depth, num_classes,
                 max_index=1, batch_size=1):
        self.feature_file = os.path.join(feature_file, "feature.csv")
        self.label_file = os.path.join(label_file, "label.csv")
        self.image_width = image_width
        self.image_height = image_height
        self.image_depth = image_depth
        self.num_classes = num_classes
        self.max_index = max_index
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        """
        The number of batches in a epoch.
        """
        return int(np.ceil(self.max_index / float(self.batch_size)))

    def __getitem__(self, index):
        """
        Generate one batch of data at 'index', which is the position of the batch in the Sequence.
        """
        # Index starts from 0
        index = index + 1
        full_index = index * self.batch_size

        rows = min(self.batch_size, self.max_index)
        if full_index > self.max_index:
            rows = self.max_index - full_index + self.batch_size
        #print("index: {}; full_index: {}; rows: {}".format(index, full_index, rows))

        # Generate data
        X, y = self.__data_generation(rows)

        return X, y

    def on_epoch_end(self):
        """
        Update after each epoch.
        """
        self.reader1 = csv.reader(open(self.feature_file, "r"))
        self.reader2 = csv.reader(open(self.label_file, "r"))

        # Skip the header row
        next(self.reader1)
        next(self.reader2)

    def __data_generation(self, rows):
        """
        Generates data containing batch_size samples
        """
        samples = np.zeros((rows, self.image_width, self.image_height, self.image_depth))
        targets = np.zeros((rows, self.image_width, self.image_height, self.num_classes))

        for j in range(rows):
            try:
                row1 = next(self.reader1)
                row2 = next(self.reader2)
            except StopIteration:
                print("CSV iteration end; calling 'break'.")
                break

            array_row1 = np.array(row1, dtype=np.float)
            samples[j,:,:,:] = preprocess_feature(array_row1,
                                                      self.image_width, self.image_height, self.image_depth)

            array_row2 = np.array(row2, dtype=np.int)
            targets[j,:,:,:] = preprocess_label(array_row2,
                                                    self.image_width, self.image_height, self.num_classes)

        return samples, targets

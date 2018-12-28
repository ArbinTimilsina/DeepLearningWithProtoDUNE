import os
import sys
import argparse
import numpy as np
import configparser
from keras.layers import Input
from keras.utils import plot_model
from keras.optimizers import Adam, SGD
from tools.data_tools import DataSequence
from tools.plotting_tools import plot_history
from tools.model_tools import get_unet_model, get_fcn_model, train_model
from tools.loss_metrics_tools import weighted_categorical_crossentropy, focal_loss

# Needed when using single GPU with sbatch; else will get the following error
# failed call to cuInit: CUDA_ERROR_NO_DEVICE
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def argument_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--operation", required=True,
	   help="Choose operation between 'Training' or 'Development.'")
    ap.add_argument("-e", "--epoch", required=True,
	   help="Options are 'Default' or a number.")
    return vars(ap.parse_args())

def main():
    config = configparser.ConfigParser()
    config_path = os.path.join("configurations", "master_configuration.ini")
    config.read(config_path)
    print("\nReading info from configuration:")

    args = argument_parser()
    if args["operation"] == "Training":
        print("\nRunnning in training setting!\n")

        NUM_TRAINING = int(config["TRAINING"]["NUM_TRAINING"])
        NUM_VALIDATION = int(config["TRAINING"]["NUM_VALIDATION"])
        NUM_EPOCHS = int(config["TRAINING"]["NUM_EPOCHS"])

    elif args["operation"] == "Development":
        print("\nRunnning in development setting!\n")

        NUM_TRAINING = int(config["DEVELOPMENT"]["NUM_TRAINING"])
        NUM_VALIDATION = int(config["DEVELOPMENT"]["NUM_VALIDATION"])
        NUM_EPOCHS = int(config["DEVELOPMENT"]["NUM_EPOCHS"])

    else:
        print("\nError: Operation should be either 'Training' or 'Development'")
        print("Exiting!\n")
        sys.exit(1)

    if args["epoch"] != "Default":
        try:
            NUM_EPOCHS = int(args["epoch"])
        except ValueError:
            print("\nError: Epoch should be an integer.")
            print("Exiting!\n")
            sys.exit(1)

    BATCH_SIZE = int(config["DEFAULT"]["BATCH_SIZE"])
    IMAGE_WIDTH = int(config["DEFAULT"]["IMAGE_WIDTH"])
    IMAGE_HEIGHT = int(config["DEFAULT"]["IMAGE_HEIGHT"])
    IMAGE_DEPTH = int(config["DEFAULT"]["IMAGE_DEPTH"])
    CLASS_NAMES = config["DEFAULT"]["CLASS_NAMES"].split()
    FEATURE_FILE_TRAINING = config["DEFAULT"]["FEATURE_FILE_TRAINING"]
    LABEL_FILE_TRAINING = config["DEFAULT"]["LABEL_FILE_TRAINING"]
    FEATURE_FILE_VALIDATION = config["DEFAULT"]["FEATURE_FILE_VALIDATION"]
    LABEL_FILE_VALIDATION = config["DEFAULT"]["LABEL_FILE_VALIDATION"]
    WEIGHTS = np.array(list(map(float, config["DEFAULT"]["WEIGHTS"].split())))

    print("NUM_TRAINING: {}".format(NUM_TRAINING))
    print("NUM_VALIDATION: {}".format(NUM_VALIDATION))
    print("NUM_EPOCHS: {}".format(NUM_EPOCHS))
    print("BATCH_SIZE: {}".format(BATCH_SIZE))
    print("IMAGE_WIDTH: {}".format(IMAGE_WIDTH))
    print("IMAGE_HEIGHT: {}".format(IMAGE_HEIGHT))
    print("IMAGE_DEPTH: {}".format(IMAGE_DEPTH))
    print("CLASS_NAMES: {}".format(CLASS_NAMES))
    print("FEATURE_FILE_TRAINING: {}".format(FEATURE_FILE_TRAINING))
    print("LABEL_FILE_TRAINING: {}".format(LABEL_FILE_TRAINING))
    print("FEATURE_FILE_VALIDATION: {}".format(FEATURE_FILE_VALIDATION))
    print("LABEL_FILE_VALIDATION: {}".format(LABEL_FILE_VALIDATION))
    print("WEIGHTS: {}".format(WEIGHTS))
    print()

    datasequence_training = DataSequence(feature_file=FEATURE_FILE_TRAINING,
                                         label_file=LABEL_FILE_TRAINING,
                                         image_width=IMAGE_WIDTH,
                                         image_height=IMAGE_HEIGHT,
                                         image_depth=IMAGE_DEPTH,
                                         num_classes=len(CLASS_NAMES),
                                         max_index=NUM_TRAINING,
                                         batch_size=BATCH_SIZE)

    datasequence_validation = DataSequence(feature_file=FEATURE_FILE_VALIDATION,
                                           label_file=LABEL_FILE_VALIDATION,
                                           image_width=IMAGE_WIDTH,
                                           image_height=IMAGE_HEIGHT,
                                           image_depth=IMAGE_DEPTH,
                                           num_classes=len(CLASS_NAMES),
                                           max_index=NUM_VALIDATION,
                                           batch_size=BATCH_SIZE)

    # Compile the model
    input_tensor = Input((IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH))
    model = get_fcn_model(input_tensor=input_tensor, num_classes=len(CLASS_NAMES), num_filters=16)

    # Plot the model architecture 
    model_path = os.path.join("plots", "model.pdf")
    plot_model(model, to_file=model_path)
    model_path_with_shapes = os.path.join("plots", "model_with_shapes.pdf")
    plot_model(model, to_file=model_path_with_shapes, show_shapes=True)

    # Different options
    test = 0
    if test == 1:
        model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
    elif test == 2:
        model.compile(optimizer=SGD(), loss=focal_loss(), metrics=['accuracy'])
    elif test == 3:
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    elif test == 4:
        model.compile(optimizer=Adam(), loss=focal_loss(), metrics=['accuracy'])
    elif test == 5:
        model.compile(optimizer=SGD(lr=1e-5, decay=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    elif test == 6:
        model.compile(optimizer=SGD(lr=1e-5, decay=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    elif test == 7:
        model.compile(optimizer=SGD(lr=1e-5, decay=1e-5, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    elif test == 8:
        model.compile(optimizer=SGD(lr=1e-5, decay=1e-5, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        print("\nError: Test is not in the range.")
        print("Exiting!\n")
        sys.exit(1)

    model_and_weights = os.path.join("saved_models", "model_and_weights.hdf5")

    # If weights exist, load them before continuing training
    continue_training = False
    if(os.path.isfile(model_and_weights) and continue_training):
        print("Old weights found!")
        try:
            model.load_weights(model_and_weights)
            print("Old weights loaded successfully!")
        except:
            print("Old weights couldn't be loaded successfully, will continue!")

    # Traing the model
    history = train_model(model=model,
                          X=datasequence_training, y=datasequence_validation,
                          num_training=NUM_TRAINING, num_validation=NUM_VALIDATION,
                          model_path=model_and_weights, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

    # Plot the history
    loss_path = os.path.join("plots", "loss_vs_epoch.pdf")
    plot_history(history, quantity='loss', plot_title='Loss', y_label='Loss', plot_name=loss_path)

    accuracy_path = os.path.join("plots", "accuracy_vs_epoch.pdf")
    plot_history(history, quantity='acc', plot_title='Accuracy', y_label='Accuracy', plot_name=accuracy_path)

if __name__ == "__main__":
    main()

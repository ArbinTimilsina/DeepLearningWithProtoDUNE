import os
import sys
import glob
import argparse
import numpy as np
import configparser
from keras.layers import Input
from keras.utils import plot_model
from keras.optimizers import RMSprop, SGD
from tools.data_tools import DataSequence
from tools.loss_metrics_tools import mean_iou
from tools.tiramisu_model import get_tiramisu_model
from tools.prediction_history import PredictionHistory
from tools.plotting_tools import plot_history, plot_feature_label_prediction
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tools.loss_metrics_tools import weighted_categorical_crossentropy, focal_loss, dice_coef_loss

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

    # Create the model
    model = get_tiramisu_model(input_tensor=input_tensor, num_classes=len(CLASS_NAMES))

    model_and_weights = os.path.join("saved_models", "model_and_weights.hdf5")

    continue_training = False
    # If weights exist, load them before continuing training
    if os.path.isfile(model_and_weights) and continue_training:
        print("Old weights found!")
        try:
            model.load_weights(model_and_weights)
            print("Old weights loaded successfully!")
        except:
            print("Old weights couldn't be loaded successfully, will continue!")

    learning_rate = 1.0e-6;
    #decaly_rate = learning_rate/NUM_EPOCHS
    #print("Decay rate is set to {}.".format(decaly_rate))

    model.compile(optimizer=RMSprop(lr=learning_rate), loss=dice_coef_loss(), metrics=['accuracy', mean_iou])

    # Print model summary
    model.summary()

    # Plot the model architecture
    model_path = os.path.join("plots", "model.pdf")
    plot_model(model, to_file=model_path, show_shapes=True)

    # Traing the model
    # Stop training when a monitored quantity has stopped improving after certain epochs
    early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=15, verbose=1)

    # Reduce learning rate when a metric has stopped improving
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.2, patience=3, cooldown=3, verbose=1)

    # Save the best model after every epoch
    check_point = ModelCheckpoint(filepath=model_and_weights, verbose=1, save_best_only=True, monitor='val_loss', mode='min')

    # To plot prediction history
    pred_history = PredictionHistory(model)

    # First clear plots- these are plotted only after training is successfully finished
    prediction_history_path = os.path.join("plots",  "prediction_history", "*.png")
    files = glob.glob(prediction_history_path)
    for prediction_history in files:
        if os.path.isfile(prediction_history):
            os.remove(prediction_history)

    loss_path = os.path.join("plots", "loss_vs_epoch.pdf")
    if os.path.isfile(loss_path):
        os.remove(loss_path)
    accuracy_path = os.path.join("plots", "accuracy_vs_epoch.pdf")
    if os.path.isfile(accuracy_path):
        os.remove(accuracy_path)
    mean_iou_path = os.path.join("plots", "mean_iou_vs_epoch.pdf")
    if os.path.isfile(mean_iou_path):
        os.remove(mean_iou_path)

    history = model.fit_generator(generator=datasequence_training,
                                  steps_per_epoch = NUM_TRAINING//BATCH_SIZE,
                                  epochs=NUM_EPOCHS,
                                  validation_data=datasequence_validation,
                                  validation_steps= NUM_VALIDATION//BATCH_SIZE,
                                  verbose=2,
                                  callbacks=[check_point, early_stop, reduce_lr, pred_history],
                                  shuffle=False,
                                  use_multiprocessing=False,
                                  workers=1)

    # Plot the history
    plot_history(history, quantity='loss', plot_title='Loss', y_label='Loss', plot_name=loss_path)
    plot_history(history, quantity='acc', plot_title='Accuracy', y_label='Accuracy', plot_name=accuracy_path)
    plot_history(history, quantity='mean_iou', plot_title='Mean IoU', y_label='Mean IoU', plot_name=mean_iou_path)

    # Plot the prediction per epoch history
    for epoch in range(NUM_EPOCHS):
        plot_feature_label_prediction_path = os.path.join("plots",  "prediction_history", "prediction_epoch_{}.png".format(epoch))
        plot_feature_label_prediction(pred_history.feature_image[epoch], pred_history.label_image[epoch],  pred_history.prediction_image[epoch],
                                      'Feature', 'Label', 'Model prediction (Epoch {})'.format(epoch), CLASS_NAMES, plot_feature_label_prediction_path)

if __name__ == "__main__":
    main()

import os
import sys
import argparse
import numpy as np
import configparser
from keras import backend as K
from keras.models import load_model
from tools.plotting_tools import plot_feature_label_prediction
from tools.loss_metrics_tools import intersection_over_union, mean_iou
from tools.data_tools import DataSequence, get_data_generator, preprocess_feature, preprocess_label
from tools.loss_metrics_tools import weighted_categorical_crossentropy, focal_loss, dice_coef_loss
from tools.loss_metrics_tools import dice_coef_loss, f_beta_score_loss

def argument_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--plots", required=True,
	   help="Choose number of events to run over to generate plots.")
    ap.add_argument("-s", "--statistics", required=True,
	   help='''Choose number of events to run over to calculate statistics.
             Options are 'Training' or 'Development'.''')
    return vars(ap.parse_args())

def mean_intersection_over_union(y_true, y_pred, class_names):
    """
    Mean Intersection over Union (IoU) over classes and batch
    """
    n_preds = y_pred.shape[0]
    print('\nNumber of validation samples IoU evaulated on: {}\n'.format(n_preds))

    total_iou = 0
    for c in range(len(class_names)):
        iou = K.eval(intersection_over_union(y_true, y_pred, c))
        print('IoU for {} is: {:.2f}%'.format(class_names[c], iou*100))
        total_iou += iou

    print('\nMean IoU is: {:.2f}%'.format(100*total_iou/len(class_names)))

def main():
    args = argument_parser()
    try:
        NUM_EVENTS_PLOTS = int(args["plots"])
        print("\nRunning over {} testing events to generate plots.".format(NUM_EVENTS_PLOTS))
    except ValueError:
        print("\nError: Events to make plots should be an integer.")
        print("Exiting!\n")
        sys.exit(1)

    config = configparser.ConfigParser()
    config_path = os.path.join("configurations", "master_configuration.ini")
    config.read(config_path)
    print("\nReading info from configuration:")

    if args["statistics"] == "Training":
        NUM_TESTING = int(config["TRAINING"]["NUM_TESTING"])
    elif args["statistics"] == "Development":
        NUM_TESTING = int(config["DEVELOPMENT"]["NUM_TESTING"])
    else:
        print("\nError: Statistics should be either 'Training' or 'Development'")
        print("Exiting!\n")
        sys.exit(1)

    print("Running over {} testing events to calculate statistics.\n".format(NUM_TESTING))

    IMAGE_WIDTH = int(config["DEFAULT"]["IMAGE_WIDTH"])
    IMAGE_HEIGHT = int(config["DEFAULT"]["IMAGE_HEIGHT"])
    IMAGE_DEPTH = int(config["DEFAULT"]["IMAGE_DEPTH"])
    CLASS_NAMES = config["DEFAULT"]["CLASS_NAMES"].split()
    WEIGHTS = np.array(list(map(float, config["DEFAULT"]["WEIGHTS"].split())))

    FEATURE_FILE_TESTING = config["DEFAULT"]["FEATURE_FILE_TESTING"]
    LABEL_FILE_TESTING = config["DEFAULT"]["LABEL_FILE_TESTING"]

    print("IMAGE_WIDTH: {}".format(IMAGE_WIDTH))
    print("IMAGE_HEIGHT: {}".format(IMAGE_HEIGHT))
    print("IMAGE_DEPTH: {}".format(IMAGE_DEPTH))
    print("CLASS_NAMES: {}".format(CLASS_NAMES))
    print("FEATURE_FILE_TESTING: {}".format(FEATURE_FILE_TESTING))
    print("LABEL_FILE_TESTING: {}".format(LABEL_FILE_TESTING))
    print("WEIGHTS: {}".format(WEIGHTS))
    print()

    # Get the model
    model_path = os.path.join("saved_models", "model_and_weights.hdf5")
    model = load_model(model_path, custom_objects={"loss": focal_loss(),
                                                   "loss": weighted_categorical_crossentropy(WEIGHTS),
                                                   "loss": dice_coef_loss(),
                                                   "loss": f_beta_score_loss(),
                                                   "mean_iou": mean_iou})

    # Make comparision plots
    generator_testing = get_data_generator(FEATURE_FILE_TESTING, LABEL_FILE_TESTING)
    count = 0
    for X, y in generator_testing:
        if count >= NUM_EVENTS_PLOTS:
            break
        count += 1

        X_preprocessed = preprocess_feature(X, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH)
        y_preprocessed = preprocess_label(y, IMAGE_WIDTH, IMAGE_HEIGHT, len(CLASS_NAMES))
        y_preprocessed_max = np.argmax(y_preprocessed, axis=3)

        prediction = model.predict_on_batch(X_preprocessed)
        prediction_max = np.argmax(prediction, axis=3)

        feature_image = X_preprocessed.reshape(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH)
        label_image = y_preprocessed_max.reshape(IMAGE_WIDTH, IMAGE_HEIGHT)
        prediction_image = prediction_max.reshape(IMAGE_WIDTH, IMAGE_HEIGHT)

        plot_feature_label_prediction_path = os.path.join("plots",  "predictions", "prediction_event_{}.pdf".format(count))
        plot_feature_label_prediction(feature_image, label_image,  prediction_image,
                                      'Feature', 'Label', 'Model prediction', CLASS_NAMES, plot_feature_label_prediction_path)

    # Calculate Statistics
    samples = np.zeros((NUM_TESTING, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH))
    targets = np.zeros((NUM_TESTING, IMAGE_WIDTH, IMAGE_HEIGHT, len(CLASS_NAMES)))

    generator_testing = get_data_generator(FEATURE_FILE_TESTING, LABEL_FILE_TESTING)
    count = 0
    for X, y in generator_testing:
        if count >= NUM_TESTING:
            break
        samples[count,:,:,:] = preprocess_feature(X, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH)
        targets[count,:,:,:] = preprocess_label(y, IMAGE_WIDTH, IMAGE_HEIGHT, len(CLASS_NAMES))
        count += 1

    predictions = model.predict_on_batch(samples)
    mean_intersection_over_union(targets, predictions, CLASS_NAMES)

    # Print the test accuracy
    score = model.evaluate(samples, targets, verbose=0)
    accuracy = 100*score[1]
    print('\nTest accuracy of the model is: {:.2f}%'.format(accuracy))

    print("\nDone!\n")

if __name__ == "__main__":
    main()

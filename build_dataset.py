# import the necessary packages
from imutils import paths
import dheeba.config.config as config
import numpy as np
import argparse
import cv2
import os

# construct the argument parser to parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required = True, help = "path to the output directory")
args = vars(ap.parse_args())

def serialize_dataset(image_paths, IMAGES_PATH, LABELS_PATH, verbose = 1000):
    print("[INFO] Building {}".format(IMAGES_PATH.split(os.path.sep)[-1]))

    images = []
    labels = []
    # loop over the image_paths
    for (i, image_path) in enumerate(image_paths):
        # extract the class label from the image_path
        label = image_path.split(os.path.sep)[-2]

        # read the image to RAM
        image = cv2.imread(image_path)
        image = cv2.resize(image, (200, 150))

        # append the image to the images list and label to the labels list
        images.append(image)
        labels.append(label)

        # print updates every verbose iterations
        if verbose > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(image_paths)))

    # serialize to disk
    print("[INFO] Saving to {} and {}".format(IMAGES_PATH, LABELS_PATH))
    np.save(IMAGES_PATH, np.array(images))
    np.save(LABELS_PATH, np.array(labels))

# grab the image paths to the datasets
train_paths = list(paths.list_images(config.TRAIN_PATH))
val_paths = list(paths.list_images(config.VAL_PATH))
test_paths = list(paths.list_images(config.TEST_PATH))

# build the output paths to the datasets
TRAIN_IMAGES_OUTPUT_PATH = os.path.sep.join([args["output"], "train_images.npy"])
TRAIN_LABELS_OUTPUT_PATH = os.path.sep.join([args["output"], "train_labels.npy"])
VAL_IMAGES_OUTPUT_PATH = os.path.sep.join([args["output"], "val_images.npy"])
VAL_LABELS_OUTPUT_PATH = os.path.sep.join([args["output"], "val_labels.npy"])
TEST_IMAGES_OUTPUT_PATH = os.path.sep.join([args["output"], "test_images.npy"])
TEST_LABELS_OUTPUT_PATH = os.path.sep.join([args["output"], "test_labels.npy"])

# # serialize the datasets to disk
serialize_dataset(train_paths, TRAIN_IMAGES_OUTPUT_PATH, TRAIN_LABELS_OUTPUT_PATH)
# serialize_dataset(val_paths, VAL_IMAGES_OUTPUT_PATH, VAL_LABELS_OUTPUT_PATH)
# serialize_dataset(test_paths, TEST_IMAGES_OUTPUT_PATH, TEST_LABELS_OUTPUT_PATH)

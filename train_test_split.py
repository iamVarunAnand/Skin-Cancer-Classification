# import the necessary packages
from sklearn.model_selection import train_test_split
from imutils import paths
import dheeba.config.config as config
import pandas as pd
import numpy as np
import argparse
import shutil
import os

# construct the argument parser to parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--csv", required = True, help = "path to the csv files")
args = vars(ap.parse_args())

# build the paths to the CSV files
VAL_CSV_PATH = os.path.sep.join([args["csv"], "HAM10000_val.csv"])
TEST_CSV_PATH = os.path.sep.join([args["csv"], "HAM10000_test.csv"])

# read the csv files into dataframes
val_df = pd.read_csv(VAL_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)

# grab the relative image paths
rel_val_paths = val_df["image_id"]
rel_test_paths = test_df["image_id"]

# grab the class labels
val_labels = val_df["dx"]
test_labels = test_df["dx"]

# construct the full paths to the images
val_paths = []
test_paths = []
for rel_val_path in rel_val_paths:
    val_paths.append(os.path.sep.join([config.ORIG_PATH, rel_val_path + ".jpg"]))

for rel_test_path in rel_test_paths:
    test_paths.append(os.path.sep.join([config.ORIG_PATH, rel_test_path + ".jpg"]))


# construct the dataset
dataset = [
            # ("training", train_paths, config.TRAIN_PATH),
            ("validation", val_paths, val_labels, config.VAL_PATH),
            ("testing", test_paths, test_labels, config.TEST_PATH)
]

# loop over the datasets
for (d_type, input_paths, labels, base_path) in dataset:
    # display which dataset is being created
    print("[INFO] building {} split".format(d_type))

    # if the base output path doesnt exist, create it
    if not os.path.exists(base_path):
        print("[INFO] creating directory {}".format(base_path))
        os.makedirs(base_path)

    # loop over the input paths
    for (i, (input_path, label)) in enumerate(zip(input_paths, labels)):
        # extract the filename and the label name from the input path
        filename = input_path.split(os.path.sep)[-1]

        # build the path to the label directory
        label_path = os.path.sep.join([base_path, label])

        # if the label directory doesnt exist, create it
        if not os.path.exists(label_path):
            print("[INFO] creating directory {}".format(label_path))
            os.makedirs(label_path)

        # construct the output image path and copy the image
        output_path = os.path.sep.join([label_path, filename])
        shutil.copy2(input_path, output_path)

        # print status to the user
        print("[INFO] processed {} / {}".format(i + 1, len(input_paths)))

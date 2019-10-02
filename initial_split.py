# import the necessary packages
from sklearn.model_selection import train_test_split
import dheeba.config.config as config
import pandas as pd
import argparse
import os

# construct the argument parser to parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True, help = "path to cleaned csv file")
ap.add_argument("-o", "--output", required = True, help = "path to output split csv files")
args = vars(ap.parse_args())

# read the csv file and extract the lesion labels
print("[INFO] reading the csv file...")
df = pd.read_csv(args["input"])
lesion_labels = pd.DataFrame(df["dx"], columns = ["dx"])

# split the dataframe into the train, validation and test sets
print("[INFO] generating the splits")
(train_df, test_df, train_labels, test_labels) = train_test_split(df, lesion_labels, test_size = config.TEST_SIZE, random_state = 42, stratify = lesion_labels)
(train_df, val_df, train_labels, val_labels) = train_test_split(train_df, train_labels, test_size = config.VAL_SIZE, random_state = 42, stratify = train_labels)

# build the paths to the output csv files
TRAIN_PATH = os.path.sep.join([args["output"], "HAM10000_train.csv"])
VAL_PATH = os.path.sep.join([args["output"], "HAM10000_val.csv"])
TEST_PATH = os.path.sep.join([args["output"], "HAM10000_test.csv"])

# serialize the dataframes to disk
print("[INFO] serializing to disk...")
train_df.to_csv(TRAIN_PATH)
val_df.to_csv(VAL_PATH)
test_df.to_csv(TEST_PATH)

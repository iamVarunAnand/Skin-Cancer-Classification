# import the necessary packages
from dheeba.utils import FindClassWeights
from dheeba.preprocessing import DataAugmenter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import cv2

# construct the argument parser to parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "path to dataset images")
ap.add_argument("-c", "--csv", required = True, help = "path to the dataset's csv description file")
ap.add_argument("-o", "--output", required = True, help = "path to store the augmented images")
args = vars(ap.parse_args())

# load the csv file from disk and calculate the class weights for augmentation
df = pd.read_csv(args["csv"])
class_weights = FindClassWeights.process(df)

# initialize the data augmenter
da = DataAugmenter(df, class_weights)
da.process(input_path = args["dataset"], output_path = args["output"])

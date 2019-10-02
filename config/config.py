# import the necessary packages
import os

# define the path to the augmented images
ORIG_PATH = "/home/varun/datasets/image_classification/HAM10000/images"

# define the base path to the split dataset
BASE_PATH = "/home/varun/datasets/image_classification/HAM10000/splits"

# derive the training, testing and validation splits from the base path
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

# define the amount of data that will be used in the various splits
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1

# define model hyperparameters
LR = 0.01
BATCH_SIZE = 8
EPOCHS = 50

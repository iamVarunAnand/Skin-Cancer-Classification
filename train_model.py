# import the necessary packages
from pyimagesearch.callbacks import TrainingMonitor
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from dheeba.nn.conv import SmallVGGNet
from imutils import paths
from dheeba.config import config
import tensorflow as tf
import argparse
import os

# initialize the tf session to allow dynamic memory growth
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess = tf.Session(config = tf_config)
set_session(sess)

# construct an argument parser to parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", required = True, help = "path to store output loss/accuracy plot")
args = vars(ap.parse_args())

# grab the image paths of the various datasets
train_paths = list(paths.list_images(config.TRAIN_PATH))
val_paths = list(paths.list_images(config.VAL_PATH))
# test_paths = list(paths.list_images(config.TEST_PATH))

# calculate the total samples in each sets
train_size = len(train_paths)
val_size = len(val_paths)
# test_size = len(test_paths)

# get the class labels from the image paths
train_labels = [path.split(os.path.sep)[-2] for path in train_paths]
val_labels = [path.split(os.path.sep)[-2] for path in val_paths]
# test_labels = [path.split(os.path.sep)[-2] for path in test_paths]

# convert the labels from string to vectors
lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
val_labels = lb.fit_transform(val_labels)

# initialize the train and val data augmenter (normalizer)
train_aug = ImageDataGenerator(rescale = 1 / 255.0)
val_aug = ImageDataGenerator(rescale = 1 / 255.0)

# initialize the train and val data generators
train_gen = train_aug.flow_from_directory(config.TRAIN_PATH, class_mode = "categorical", target_size = (150, 200), color_mode = "rgb", shuffle = True, batch_size=config.BATCH_SIZE)

val_gen = val_aug.flow_from_directory(config.VAL_PATH, class_mode = "categorical", target_size = (150, 200), color_mode = "rgb", shuffle = False, batch_size = config.BATCH_SIZE)

# initialize callbacks
fig_path = os.path.sep.join([args["plot"], "expt-1.png"])
callbacks = [TrainingMonitor(fig_path)]

# initialize the model and compile it
model = SmallVGGNet.build(150, 200, 3, 7)
opt = SGD(lr = config.LR, momentum = 0.9, nesterov = True)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

# train the model
H = model.fit_generator(train_gen,
                        steps_per_epoch = train_size // config.BATCH_SIZE,
                        validation_data = val_gen,
                        validation_steps = val_size // config.BATCH_SIZE,
                        epochs = config.EPOCHS)

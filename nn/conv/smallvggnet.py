# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers import Input
from keras import backend as K

class SmallVGGNet:
    @staticmethod
    def build(height, width, depth, classes):
        # set the input shape to match the channel ordering
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            channel_dim = 1
        else:
            input_shape = (height, width, depth)
            channel_dim = -1

        # instantiate the model
        model = Sequential()

        # first (and only) CONV => RELU => POOL block
        model.add(Conv2D(32, (3, 3), padding = "same", input_shape = input_shape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = channel_dim))
        model.add(MaxPooling2D(pool_size = (3, 3)))
        model.add(Dropout(0.25))

        # first CONV => RELU => CONV => RELU => POOL block
        model.add(Conv2D(64, (3, 3), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = channel_dim))
        model.add(Conv2D(64, (3, 3), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = channel_dim))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        # second CONV => RELU => CONV => RELU => POOL block
        model.add(Conv2D(128, (3, 3), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = channel_dim))
        model.add(Conv2D(128, (3, 3), padding = "same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = channel_dim))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        # first (and only) FC layer
        model.add(Flatten())
        model.add(Dense(32))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = channel_dim))
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model

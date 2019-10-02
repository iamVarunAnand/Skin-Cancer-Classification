# import the necessary packages
import numpy as np

class FindClassWeights:
    @staticmethod
    def process(df):
        # initialize a dictionary to store the number of instances of each class in the dataset
        class_counts = {'nv': 0, 'mel': 0, 'bkl': 0,
                        'bcc': 0, 'akiec': 0, 'vasc': 0,
                        'df': 0}

        # loop through the classes and store their corresponding counts
        for key in class_counts.keys():
            count = df.groupby("dx")["dx"].count()[key]
            class_counts[key] = count

        # initialize the class weights dictionary
        class_weights = {'nv': 0, 'mel': 0, 'bkl': 0,
                        'bcc': 0, 'akiec': 0, 'vasc': 0,
                        'df': 0}

        # find the sample count of the class with the most samples
        max_count = np.max(list(class_counts.values()))

        # loop through the classes to find their corresponding weights
        for key in class_counts.keys():
            class_weights[key] = np.floor(max_count / class_counts[key])

        return class_weights

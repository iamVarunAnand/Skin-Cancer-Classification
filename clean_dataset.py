# import the necessary packages
import pandas as pd
import numpy as np
import argparse

# construct the argument parser to parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True, help = "path to the dataset's csv description file")
ap.add_argument("-o", "--output", required = True, help = "path to store the cleaned csv description file")
args = vars(ap.parse_args())

def extract_unique(df, verbose = 1000):
    # initialize a dictionary to store the current count of the corresponding lesion id
    lid_counts = {}

    # initialize the new unique dataframe
    unique_df = pd.DataFrame()

    # loop through the tuples in the dataframe
    for i in range(df.shape[0]):
        # get the current count of the corresponding lid
        lid = df.iloc[i]["lesion_id"]
        count = lid_counts.get(lid, 0);

        # if the count is 0, add the corresponding lesion to the new dataframe
        if(count == 0):
            unique_df = unique_df.append(df.iloc[i, :])

        # increment the count corresponding to the current lid
        count = count + 1
        lid_counts[lid] = count

        if(verbose > 0 and (i + 1) % verbose == 0):
            print("[INFO] processed {}/{} tuples".format(i + 1, df.shape[0]))

    # return the dataframe with unique lesion ids
    return unique_df

def main():
    # load the csv file
    data = pd.read_csv(args["input"])

    # remove duplicate lesions from the dataset
    unique_data = extract_unique(data)

    # save the dataframe to disk
    unique_data.to_csv(args["output"])

if __name__ == "__main__":
    main()

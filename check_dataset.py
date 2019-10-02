# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import cv2

# load the images
print("[INFO] loading the dataset...")
images = np.load("dataset/train_images.npy")
print(images.shape)

# generate 4 random indices
indices = np.random.randint(0, images.shape[0], size = (4,))

# plot the images
plt.figure()
for (c, i) in enumerate(indices):
    plt.subplot(2, 2, c + 1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
plt.show()

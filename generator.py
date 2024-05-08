import os.path
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import math

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.count = 0
        self.epoch = 0

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases

        with open(self.label_path) as label_file:
            label_content = json.load(label_file)

        image_indeces = np.arange(len(label_content))

        images = []
        labels = []

        if self.shuffle:
            np.random.shuffle(image_indeces)

        # For last batch in the epoch
        if (self.count + 1) * self.batch_size > len(label_content):
            offset = ((self.count + 1) * self.batch_size) - len(label_content)
            current_batch = image_indeces[self.count * self.batch_size : len(label_content)]
            current_batch = np.append(current_batch, image_indeces[0: offset])
            self.count = -1 # In order to reset count value
            self.epoch += 1 # to count epoch number
        else:
            current_batch = image_indeces[self.count * self.batch_size : (self.count + 1) * self.batch_size]

        #Collecting data from the source
        for i in current_batch:
            images.append(np.load(os.path.join(self.file_path, str(i) + '.npy')))
            labels.append(label_content[str(i)])

        #Resizing the source images
        for i, image in enumerate(images):
            images[i] = resize(image, self.image_size)

        #If the Mirroring and rotation is True
        for i, image in enumerate(images):
            images[i] = self.augment(image)

        #list to np array
        labels = np.asarray(labels)
        images = np.asarray(images)
        self.count = self.count + 1 #Count increase to count batch

        return images, labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function

        if self.mirroring:
            num = np.random.randint(0, 2, 1)
            if num[0] == 1:
                img = np.fliplr(img)

        if self.rotation:
            num = np.random.randint(0, 4, 1)
            num = num[0]
            img = np.rot90(img, num)

        return img

    def current_epoch(self):
        epoch = self.epoch
        return epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.class_dict[x]
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        images, labels = self.next()

        for i, image in enumerate(images):
            if self.batch_size > 3:
                num_of_row = math.ceil(self.batch_size/3)
            else:
                num_of_row = 1

            plt.subplot(num_of_row, 3, i+1)
            plt.title(self.class_name(labels[i]))
            plt.axis('off')
            plt.imshow(image)
        plt.show()


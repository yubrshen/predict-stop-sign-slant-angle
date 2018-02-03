from __future__ import division

import math
import random
import os
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.preprocessing.image import Iterator
from keras.utils.np_utils import to_categorical
import keras.backend as K

max_range = 80

def deg_f(file_name):
    """
    Extract the degree component from the file_name.
    Returns a string of the degree.
    """
    pattern = re.compile(r'.*sign_([-]?\d+)_\d+\.jpg')
    matched = pattern.match(file_name)
    return matched.group(1)

def get_filenames(path, ratio=(0.7, 0.2, 0.1)):
    """
    Extract the file names in the path (a directory), split the file names into
    3 groups of train_filenames, validation_filenames, and test_filenames
    according to the ratio for each group.

    The ration, as illustrated by the default value, is an array of 3 elements
    adding to 1 for train, validation, and test respectively.

    The ratio of the split should be applied uniformly for all degrees.

    The order of the filenames within a group has been shuffled already.
    The order of the filenames merged amoung degree groups returned
    have also been re-shuffled.

    Return the 3 filename lists.

    """
    # group the filenames by degree
    sample_groups_by_degree = {}
    for filename in os.listdir(path):
        deg = deg_f(filename)
        group = sample_groups_by_degree.get(deg, [])
        group.append(os.path.join(path, filename))
        sample_groups_by_degree[deg] = group

    # split per groups and assemble
    # train_filenames = []
    # validation_filenames = []
    # test_filenames = []
    filename_groups = [[]]*len(ratio)
    for deg, samples in sample_groups_by_degree.items():
        random.shuffle(samples)
        cursor = 0
        for i in range(len(ratio)):
            end = int(cursor+len(samples)*ratio[i])
            filename_groups[i] = filename_groups[i] + (samples[cursor:end])
            cursor = end

    # re-shuffle among the degree groups
    for filenames in filename_groups:
        random.shuffle(filenames)

    return filename_groups


class DataGenerator(Iterator):
    """
    Given a NumPy array of images or a list of image paths,
    generate batches of images and with the corresponding labels
    (target values)
    """

    def __init__(self, input, targets=None, input_shape=None, color_mode='rgb',
               batch_size=64, one_hot=True, preprocess_func=None, shuffle=False, seed=None):

        self.images = None
        self.filenames = None
        self.input_shape = input_shape
        self.color_mode = color_mode
        self.batch_size = batch_size
        self.one_hot = one_hot
        self.preprocess_func = preprocess_func
        self.shuffle = shuffle

        if self.color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', self.color_mode,
                             '; expected "rgb" or "grayscale".')

        # check whether the input is a NumPy array or a list of paths
        if isinstance(input, (np.ndarray)):
            self.images = input
            N = self.images.shape[0]
            if not self.input_shape:
                self.input_shape = self.images.shape[1:]
                # add dimension if the images are greyscale
                if len(self.input_shape) == 2:
                    self.input_shape = self.input_shape + (1,)
            self.targets = targets
            # when the input is Numpy array, so is the targets
        else:
            self.filenames = input
            N = len(self.filenames)

        super(DataGenerator, self).__init__(N, batch_size, shuffle, seed)

    def _get_batches_of_samples(self, index_array):
        # create array to hold the images
        batch_x = np.zeros((len(index_array),) + self.input_shape, dtype='float32')
        # create array to hold the labels
        batch_y = np.zeros(len(index_array), dtype='float32')

        # iterate through the current batch
        for i, j in enumerate(index_array):
            if self.filenames is None:
                image = self.images[j]
                target = self.targets[j]
            else:
                is_color = int(self.color_mode == 'rgb')
                image = cv2.imread(self.filenames[j], is_color)
                if is_color:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                target = int(deg_f(self.filenames[j]))

            # store the image and label in their corresponding batches
            batch_x[i] = image
            batch_y[i] = target

        if self.one_hot:
            # convert the numerical labels to binary labels
            nb_classes = 161
            batch_y = to_categorical(batch_y, nb_classes)
        else:
            batch_y /= max_range
            # The maximum asbsolute value is 80

        # preprocess input images
        if self.preprocess_func:
            batch_x = self.preprocess_func(batch_x)

        return batch_x, batch_y

    def next(self):
        with self.lock:
            # get input data index and size of the current batch
            index_array, _, current_batch_size = next(self.index_generator)
        # create array to hold the images
        return self._get_batches_of_samples(index_array)


def display_examples(model, input, num_images=5, size=[275, 275],
                   preprocess_func=None, save_path=None):
    """
    Given a model that predicts the slant tilt angle of a sotp sign
    in an image,
    and a  list of image paths, display
    the specified number of example images
    with the original angles and the predicted angles shown.
    """
    images = []
    slant_degrees = []
    filenames = input
    N = len(filenames)
    indexes = np.random.choice(N, num_images)
    for i in indexes:
        # image = mpimg.imread(filenames[i])
        image = cv2.imread(filenames[i], 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #.astype('float32')
        images.append(image)
        slant_degrees.append(int(deg_f(filenames[i])))
    slant_degrees = np.asarray(slant_degrees, dtype='float32')

    if preprocess_func:
        images_processed = preprocess_func(np.asarray(images, dtype='float32'))

    degrees_pred = np.squeeze(model.predict(images_processed))*max_range

    plt.figure(figsize=(10.0, 2 * num_images))

    title_fontdict = {
        'fontsize': 14,
        'fontweight': 'bold'
    }

    fig_number = 0
    for image, true_angle, predicted_angle in zip(images, slant_degrees, degrees_pred):
        fig_number += 1
        ax = plt.subplot(num_images, 3, fig_number)
        plt.title('True angle: {:.0f}\nPredicted angle: {:.2f}'.format
                  (true_angle, predicted_angle), fontdict=title_fontdict)
        plt.imshow(image)
        plt.axis('off')

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()

    if save_path:
        plt.savefig(save_path)


print("utils.py loaded")

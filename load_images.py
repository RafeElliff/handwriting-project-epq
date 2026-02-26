import os
import numpy
import time
import json
import tensorflow_datasets
import random
from helper_functions import scale_array_to_0_to_1
import cv2

base_maths = r"C:\Users\rafee\PycharmProjects\data-for-handwriting-epq"

def get_maths_images(starting_index, finishing_index, training_or_testing):
    #The maths images were all downloaded as one: I had to manually split them into a 'training' and 'testing' file.
    batch_size = finishing_index - starting_index
    if training_or_testing == "training":
        images_in_folder = os.listdir(os.path.join(base_maths, "training_skeletons")) #This gets all of the names of skeletons in the folder
        images_in_folder.sort(key=lambda x: int(x.split('.')[0])) #Lists them in ascending order by their ids. This seems like it would not be necessary, but forgoing it can lead to unexpected results
        images_to_add = images_in_folder[starting_index:finishing_index] #Gets only the desired images
        list_of_numpys = []
        for image_name in images_to_add: #Loads images 1 by 1 into a list which wil eventually get stacked into mata matrix
            image_numpy = numpy.load(os.path.join(base_maths, "training_skeletons", image_name))
            list_of_numpys.append(image_numpy)
        images_matrix = numpy.stack(list_of_numpys)
        images_matrix = images_matrix.reshape(batch_size, 28, 28, 1)
        with open(os.path.join(base_maths, "training_labels.json"), "r") as file: #This is a json file where each label corresponds to an image, e.g. the first label would be the label for image 1 in the training_skeletons folder
            labels = json.load(file)
            labels_sliced = labels[starting_index: finishing_index]
            labels_matrix = numpy.stack(labels_sliced)
    elif training_or_testing == "testing":# The code here is functionally identical to the above block but pulls from the testing set instead
        images_in_folder = os.listdir(os.path.join(base_maths, "testing_skeletons"))
        images_in_folder.sort(key=lambda x: int(x.split('.')[0]))
        images_to_add = images_in_folder[starting_index:finishing_index]
        list_of_numpys = []
        for image_name in images_to_add:
            image_numpy = numpy.load(os.path.join(base_maths, "testing_skeletons", image_name))
            list_of_numpys.append(image_numpy)
        images_matrix = numpy.stack(list_of_numpys)
        images_matrix = images_matrix.reshape(batch_size, 28, 28, 1)
        with open(os.path.join(base_maths, "testing_labels.json"), "r") as file:
            labels = json.load(file)
            labels_sliced = labels[starting_index: finishing_index]
            labels_matrix = numpy.stack(labels_sliced)
    return images_matrix, labels_matrix

def get_EMNIST_images(starting_index, finishing_index, training_or_testing):
    if training_or_testing == "training":
        training_dataset, dataset_info = tensorflow_datasets.load(
            "emnist/bymerge",
            split=f"train[{starting_index}:{finishing_index}]", #The syntax surrounding this function is a bit weird, but this just pulls a list of labels and images from the EMNIST training dateset
            as_supervised=True,
            with_info=True)
        training_images = []
        training_labels = []
        for image, label in training_dataset:
            image = numpy.array(image)[:, :, 0]
            transposed = numpy.transpose(image)
            skeleton = cv2.ximgproc.thinning(transposed) #Gets the 'skeleton' of an image: this reduces the image to a single-pixel wide line: if removing a pixel would not affect the connectivity of an image, then it will be removed.
            scaled = scale_array_to_0_to_1(skeleton, inverse=False)
            reshaped = numpy.reshape(scaled, (28, 28, 1)) #The third dimension is needed here because of how I programmed the forward pass function
            training_images.append(reshaped)
            training_labels.append(numpy.array(label))
        images_matrix = numpy.stack(training_images) #Stack converts an array of 1d matrices into a 2d matrix
        labels_matrix = numpy.stack(training_labels)
        return images_matrix, labels_matrix

    elif training_or_testing == "testing": #This is functionally identical to the block of code above but pulls from the testing dataset instead
        testing_dataset, dataset_info = tensorflow_datasets.load(
            "emnist/bymerge",
            split=f"test[{starting_index}:{finishing_index}]",
            as_supervised=True,
            with_info=True)
        testing_images = []
        testing_labels = []
        for image, label in testing_dataset:
            image = numpy.array(image)[:, :, 0]
            transposed = numpy.transpose(image)
            skeleton = cv2.ximgproc.thinning(transposed)
            scaled = scale_array_to_0_to_1(skeleton, inverse=False)
            reshaped = numpy.reshape(scaled, (28, 28, 1))
            testing_images.append(reshaped)
            testing_labels.append(numpy.array(label))

        images_matrix = numpy.stack(testing_images)
        labels_matrix = numpy.stack(testing_labels)
        return images_matrix, labels_matrix

def get_full_set(maths_starting, maths_finishing, EMNIST_starting, EMNIST_finishing, training_or_testing): #Gets an aggregate of a certain number of maths images and a certain number of EMNIST images and combines them into one dataset
    maths_images_matrix, maths_labels_matrix = get_maths_images(maths_starting, maths_finishing, training_or_testing)
    EMNIST_images_matrix, EMNIST_labels_matrix = get_EMNIST_images(EMNIST_starting, EMNIST_finishing, training_or_testing)
    final_images_matrix = numpy.concatenate([maths_images_matrix, EMNIST_images_matrix])
    final_labels_matrix = numpy.concatenate([maths_labels_matrix, EMNIST_labels_matrix])
    shuffled_indices = list(range(0, len(final_images_matrix)))
    random.shuffle(shuffled_indices) #It's beneficial to shuffle them because otherwise you'll have all of the EMNIST images then all of the maths images. This can mess with the gradient descent algorithm.
    final_labels_matrix_shuffled = final_labels_matrix[shuffled_indices] #By passing in a list of indices, you can rearrange the elements into those specified indices
    final_images_matrix_shuffled = final_images_matrix[shuffled_indices]
    return final_images_matrix_shuffled, final_labels_matrix_shuffled

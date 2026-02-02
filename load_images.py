import os
import numpy
import time
import json
import tensorflow_datasets
import random
from helper_functions import scale_array_to_0_to_1
base_maths = r"C:\Users\rafee\PycharmProjects\data-for-handwriting-epq"

##
def get_maths_images(starting_index, finishing_index, training_or_testing):
    batch_size = finishing_index - starting_index
    if training_or_testing == "training":
        images_in_folder = os.listdir(os.path.join(base_maths, "training_images"))
        images_in_folder.sort(key=lambda x: int(x.split('.')[0]))
        images_to_add = images_in_folder[starting_index:finishing_index]
        list_of_numpys = []
        for image_name in images_to_add:
            image_numpy = numpy.load(os.path.join(base_maths, "training_images", image_name))
            image_numpy = image_numpy.astype(numpy.float32)
            list_of_numpys.append(image_numpy)
        images_matrix = numpy.stack(list_of_numpys)
        images_matrix = images_matrix.reshape(batch_size, 28, 28, 1)
        with open(os.path.join(base_maths, "training_labels.json"), "r") as file:
            labels = json.load(file)
            labels_sliced = labels[starting_index: finishing_index]
            labels_matrix = numpy.stack(labels_sliced)
    elif training_or_testing == "testing":
        images_in_folder = os.listdir(os.path.join(base_maths, "testing_images"))
        images_in_folder.sort(key=lambda x: int(x.split('.')[0]))
        images_to_add = images_in_folder[starting_index:finishing_index]
        list_of_numpys = []
        for image_name in images_to_add:
            image_numpy = numpy.load(os.path.join(base_maths, "testing_images", image_name))
            image_numpy = image_numpy.astype(numpy.float32)
            list_of_numpys.append(image_numpy)
        images_matrix = numpy.stack(list_of_numpys)
        images_matrix = images_matrix.reshape(batch_size, 28, 28, 1)
        with open(os.path.join(base_maths, "testing_labels.json"), "r") as file:
            labels = json.load(file)
            labels_sliced = labels[starting_index: finishing_index]
            labels_matrix = numpy.stack(labels_sliced)
    return images_matrix, labels_matrix


# for i in range (0, 50):
#     starting_index = 50 * i
#     time_before = time.time()
#     maths_images_matrix, labels_matrix = get_maths_images(starting_index, starting_index+50, "training")
#     print(labels_matrix)
#     time_after = time.time()
#     print(time_after-time_before, i)

def get_EMNIST_images(starting_index, finishing_index, training_or_testing):
    if training_or_testing == "training":
        training_dataset, dataset_info = tensorflow_datasets.load(
            "emnist/bymerge",
            split=f"train[{starting_index}:{finishing_index}]", # Processing all of the data takes a fair bit of time. I'm choosing to only load the first few samples for now and when I get to proper training obviously I'll load more.
            as_supervised=True,
            with_info=True)
        training_images = []
        training_labels = []
        for image, label in training_dataset:
            image = numpy.array(image, dtype=numpy.float32)
            scaled = scale_array_to_0_to_1(image, inverse=False)
            transposed = numpy.transpose(scaled)
            reshaped = numpy.reshape(transposed, (28, 28, 1))
            training_images.append(reshaped)
            training_labels.append(numpy.array(label))
        images_matrix = numpy.stack(training_images)
        labels_matrix = numpy.stack(training_labels)
        return images_matrix, labels_matrix

    elif training_or_testing == "testing":
        testing_dataset, dataset_info = tensorflow_datasets.load(
            "emnist/bymerge",
            split=f"test[{starting_index}:{finishing_index}]",
            as_supervised=True,
            with_info=True)
        testing_images = []
        testing_labels = []
        for image, label in testing_dataset:
            image = numpy.array(image)
            scaled = scale_array_to_0_to_1(image, inverse=False)
            transposed = numpy.transpose(scaled)
            reshaped = numpy.reshape(transposed, (28, 28, 1))
            testing_images.append(reshaped)
            testing_labels.append(numpy.array(label))

        images_matrix = numpy.stack(testing_images)
        labels_matrix = numpy.stack(testing_labels)
        return images_matrix, labels_matrix

def get_full_set(maths_starting, maths_finishing, EMNIST_starting, EMNIST_finishing, training_or_testing):
    maths_images_matrix, maths_labels_matrix = get_maths_images(maths_starting, maths_finishing, training_or_testing)
    EMNIST_images_matrix, EMNIST_labels_matrix = get_EMNIST_images(EMNIST_starting, EMNIST_finishing, training_or_testing)
    final_images_matrix = numpy.concatenate([maths_images_matrix, EMNIST_images_matrix])
    final_labels_matrix = numpy.concatenate([maths_labels_matrix, EMNIST_labels_matrix])
    shuffled_indices = list(range(0, len(final_images_matrix)))
    random.shuffle(shuffled_indices)
    final_labels_matrix_shuffled = final_labels_matrix[shuffled_indices]
    final_images_matrix_shuffled = final_images_matrix[shuffled_indices]
    return final_images_matrix_shuffled, final_labels_matrix_shuffled

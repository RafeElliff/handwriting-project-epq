import os
import cv2
import numpy
import random
import json
import time
from helper_functions import resize_to_28_x_28, scale_array_to_0_to_1
maths_source_data = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\all_training_data\maths\maths_source_data"
twentyeight_x_twentyeight_images = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\all_training_data\maths\28_x_28_images"
base_maths = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\all_training_data\maths"


def rename_and_move_images():     #This code renames all the images to have the format {id}_{class}.npy
    image_id = -1
    for class_of_character in os.listdir(maths_source_data):
        folder = os.path.join(maths_source_data, class_of_character)
        print(folder)
        for image_name in os.listdir(folder):
            print(image_name)
            image_id = image_id + 1
            source_path = os.path.join(folder, image_name)
            image_new_name = f"{str(image_id)}_{class_of_character}"
            new_path = os.path.join(twentyeight_x_twentyeight_images, image_new_name)
            fortyfive_fortyfive_image = cv2.imread(source_path)
            gray_image = cv2.cvtColor(fortyfive_fortyfive_image, cv2.COLOR_BGR2GRAY)
            twentyeight_twentyeight_image = resize_to_28_x_28(gray_image)
            colour_scaled_image = scale_array_to_0_to_1(twentyeight_twentyeight_image)
            numpy.save(new_path, colour_scaled_image)

labels_to_numbers = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'A': 10,
    'B': 11,
    'C': 12,
    'D': 13,
    'E': 14,
    'F': 15,
    'G': 16,
    'H': 17,
    'I': 18,
    'J': 19,
    'K': 20,
    'L': 21,
    'M': 22,
    'N': 23,
    'O': 24,
    'P': 25,
    'Q': 26,
    'R': 27,
    'S': 28,
    'T': 29,
    'U': 30,
    'V': 31,
    'W': 32,
    'X': 33,
    'Y': 34,
    'Z': 35,
    'a': 36,
    'b': 37,
    'd': 38,
    'e': 39,
    'f': 40,
    'g': 41,
    'h': 42,
    'n': 43,
    'q': 44,
    'r': 45,
    't': 46,
    '!': 47,
    '(': 48,
    ')': 49,
    '+': 50,
    ',': 51,
    '-': 52,
    '=': 53,
    '[': 54,
    ']': 55,
    'alpha': 56,
    'ascii_124': 57,
    'beta': 58,
    'Delta': 59,
    'div': 60,
    'exists': 61,
    'forall': 62,
    'forward_slash': 63,
    'gamma': 64,
    'gt': 65,
    'in': 66,
    'infty': 67,
    'int': 68,
    'lambda': 69,
    'lt': 70,
    'mu': 71,
    'neq': 72,
    'phi': 73,
    'pi': 74,
    'prime': 75,
    'rightarrow': 76,
    'sigma': 77,
    'sqrt': 78,
    'sum': 79,
    'theta': 80,
    'times': 81,
    '{': 82,
    '}': 83
}


number_of_images = 234647
# list_from_0_to_n = list(range(0, number_of_images))
# random.shuffle(list_from_0_to_n)
# print(list_from_0_to_n[0:100])
# with open(os.path.join(base_maths, "shuffled_indices.json"), "w") as file:
#     json.dump(list_from_0_to_n, file)




def produce_random_ordering_and_labels():
    with open(os.path.join(base_maths, "training_labels.json"), "r") as file:
        training_labels = json.load(file)
    with open(os.path.join(base_maths, "testing_labels.json"), "r") as file:
        testing_labels = json.load(file)
    remaining_indices = check_progress()
    image_positions = os.listdir(os.path.join(base_maths, "28_x_28_images"))
    image_new_position = number_of_images - 1 - len(remaining_indices)
    time_last_cycle = time.time()
    for image_old_position in remaining_indices:
        image_new_position = image_new_position + 1
        image_old_name = image_positions[image_old_position]
        image_components = image_old_name.split("_")
        image_components_without_id = image_components[1:]
        image_label_with_suffix = "_".join(image_components_without_id)
        image_label = image_label_with_suffix[:-4]
        image_label_as_number = labels_to_numbers[image_label]
        if image_new_position % 1000 == 0:
            time_per_1k = time.time()-time_last_cycle
            print(image_new_position, image_old_position, f"Time for 1000 images: {round(time_per_1k, 3)}")
            time_last_cycle = time.time()
        if image_new_position < 208000:
            destination_path = os.path.join(base_maths, "training_images", f"{image_new_position}.npy")
            image_as_npy = numpy.load(os.path.join(twentyeight_x_twentyeight_images, image_old_name))
            numpy.save(destination_path, image_as_npy)
            training_labels.append(image_label_as_number)
        else:
            destination_path = os.path.join(base_maths, "testing_images", f"{image_new_position}.npy")
            image_as_npy = numpy.load(os.path.join(twentyeight_x_twentyeight_images, image_old_name))
            numpy.save(destination_path, image_as_npy)
            testing_labels.append(image_label_as_number)

        if image_new_position % 1000 == 0:
            with open(os.path.join(base_maths, "testing_labels.json"), "w") as file:
                json.dump(testing_labels, file)
            with open(os.path.join(base_maths, "training_labels.json"), "w") as file:
                json.dump(training_labels, file)

def check_progress():
    with open(os.path.join(base_maths, "testing_labels.json"), "r") as file:
        testing_labels = json.load(file)
    with open(os.path.join(base_maths, "training_labels.json"), "r") as file:
        training_labels = json.load(file)
    progress_labels = len(training_labels) + len(testing_labels)
    with open(os.path.join(base_maths, "shuffled_indices.json"), "r") as file:
        shuffled_indices = json.load(file)
    remaining_indices = shuffled_indices[progress_labels:]
    return remaining_indices


# rename_and_move_images()
produce_random_ordering_and_labels()
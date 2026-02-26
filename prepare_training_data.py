import os
import cv2
import numpy
import random
import json
import time
from helper_functions import resize_to_28_x_28, scale_array_to_0_to_1, view_numpy_as_png
maths_source_data = r"C:\Users\rafee\PycharmProjects\data-for-handwriting-epq\raw_images"
closed_binarised_resized = r"C:\Users\rafee\PycharmProjects\data-for-handwriting-epq\closed_binarised_resized"
skeletons = r"C:\Users\rafee\PycharmProjects\data-for-handwriting-epq\skeletons"
base_maths = r"C:\Users\rafee\PycharmProjects\data-for-handwriting-epq"
#
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
numbers_to_labels = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: 'A',
    11: 'B',
    12: 'C',
    13: 'D',
    14: 'E',
    15: 'F',
    16: 'G',
    17: 'H',
    18: 'I',
    19: 'J',
    20: 'K',
    21: 'L',
    22: 'M',
    23: 'N',
    24: 'O',
    25: 'P',
    26: 'Q',
    27: 'R',
    28: 'S',
    29: 'T',
    30: 'U',
    31: 'V',
    32: 'W',
    33: 'X',
    34: 'Y',
    35: 'Z',
    36: 'a',
    37: 'b',
    38: 'd',
    39: 'e',
    40: 'f',
    41: 'g',
    42: 'h',
    43: 'n',
    44: 'q',
    45: 'r',
    46: 't',
    47: '!',
    48: '(',
    49: ')',
    50: '+',
    51: ',',
    52: '-',
    53: '=',
    54: '[',
    55: ']',
    56: 'alpha',
    57: 'ascii_124',
    58: 'beta',
    59: 'Delta',
    60: 'div',
    61: 'exists',
    62: 'forall',
    63: 'forward_slash',
    64: 'gamma',
    65: 'gt',
    66: 'in',
    67: 'infty',
    68: 'int',
    69: 'lambda',
    70: 'lt',
    71: 'mu',
    72: 'neq',
    73: 'phi',
    74: 'pi',
    75: 'prime',
    76: 'rightarrow',
    77: 'sigma',
    78: 'sqrt',
    79: 'sum',
    80: 'theta',
    81: 'times',
    82: '{',
    83: '}'
}

#README: this code should not be run. It is a single use block of code that was used to convert the raw maths images into a format that I can work with
#This code will not run unless you have the appropriate files


def convert_maths_image_to_skeleton(file): #This code converts a maths image from its raw form into a 28*28 skeleton (the same format as the other data)
    inverted = 255 - file
    _, thresholded_at_16 = cv2.threshold(inverted, 16, 255, cv2.THRESH_BINARY)
    kernel = numpy.ones((2, 2), numpy.uint8)
    dilated = cv2.dilate(thresholded_at_16, kernel, iterations=1)
    resized = resize_to_28_x_28(dilated) #The original maths images are 45*45
    _, rethresholded = cv2.threshold(resized, 16, 255, cv2.THRESH_BINARY) #Resizing can ruin the threshold, so rethresholding is necessary.
    skeleton = cv2.ximgproc.thinning(rethresholded)
    return skeleton

def rename_and_move_images():
    image_id = -1
    #This block of code loops through every raw source image, renames it to have the format {id}_{class}.npy, and stores it in a new location
    for class_of_character in os.listdir(maths_source_data):
        folder = os.path.join(maths_source_data, class_of_character)
        print(folder, len(os.listdir(folder)))
        for image_name in os.listdir(folder):
            image_id = image_id + 1
            if image_id % 1000 == 0:
                print(image_id)
            source_path = os.path.join(folder, image_name)
            image_new_name = f"{str(image_id)}_{class_of_character}" + ".npy"
            final_path = os.path.join(skeletons, image_new_name)
            fortyfive_fortyfive_image = cv2.imread(source_path)
            gray_image = cv2.cvtColor(fortyfive_fortyfive_image, cv2.COLOR_BGR2GRAY)
            skeleton = convert_maths_image_to_skeleton(gray_image)
            colour_scaled_image = scale_array_to_0_to_1(skeleton, inverse=False )
            numpy.save(final_path, colour_scaled_image)

number_of_images = 234647

def produce_random_ordering_and_labels():
    with open(os.path.join(base_maths, "training_labels.json"), "r") as file:
        training_labels = json.load(file)
    with open(os.path.join(base_maths, "testing_labels.json"), "r") as file:
        testing_labels = json.load(file)
    remaining_indices = check_progress()
    image_positions = os.listdir(os.path.join(base_maths, "skeletons"))
    image_new_position = number_of_images - 1 - len(remaining_indices)
    time_last_cycle = time.time()
    for image_old_position in remaining_indices:
        image_new_position = image_new_position + 1
        image_old_name = image_positions[image_old_position]
        image_components = image_old_name.split("_")
        image_components_without_id = image_components[1:]
        image_label_with_suffix = "_".join(image_components_without_id)
        image_label = image_label_with_suffix[:-4]
        image_label_as_number = labels_to_numbers[image_label] #Various processing functions to get the new name
        if image_new_position % 1000 == 0: #This entire if block is just for debugging, can be commented out
            time_per_1k = time.time()-time_last_cycle
            print(image_new_position, image_old_position, f"Time for 1000 images: {round(time_per_1k, 3)}")
            time_last_cycle = time.time()
        if image_new_position < 208000: #Segments the data into a training set and a testing set
            destination_path = os.path.join(base_maths, "training_skeletons", f"{image_new_position}.npy")
            image_as_npy = numpy.load(os.path.join(skeletons, image_old_name))
            numpy.save(destination_path, image_as_npy)
            training_labels.append(image_label_as_number)
        else:
            destination_path = os.path.join(base_maths, "testing_skeletons", f"{image_new_position}.npy")
            image_as_npy = numpy.load(os.path.join(skeletons, image_old_name))
            numpy.save(destination_path, image_as_npy)
            testing_labels.append(image_label_as_number)

        if image_new_position % 1000 == 0: #See check_progress below for rationale behind this
            with open(os.path.join(base_maths, "testing_labels.json"), "w") as file:
                json.dump(testing_labels, file)
            with open(os.path.join(base_maths, "training_labels.json"), "w") as file:
                json.dump(training_labels, file)

def check_progress(): #I kept getting memory issues while running this code initially so it didn't terminate. I wrote this so I don't have to start from scratch every time.
    #Checks how many thousands of images have been processed so far, and then starts again from that index as opposed to starting from index 0 every time
    with open(os.path.join(base_maths, "testing_labels.json"), "r") as file:
        testing_labels = json.load(file)
    with open(os.path.join(base_maths, "training_labels.json"), "r") as file:
        training_labels = json.load(file)
    progress_labels = len(training_labels) + len(testing_labels)
    with open(os.path.join(base_maths, "shuffled_indices.json"), "r") as file:
        shuffled_indices = json.load(file)
    remaining_indices = shuffled_indices[progress_labels:]
    return remaining_indices




#README: this code is commented out to avoid accidentally running it, but would be used to process the maths images again.
##This code makes it so the images within the dataset are shuffled as opposed to e.g. 5000 images of "+" then 5000 images of "(" etc.
# shuffled_indices = list(range(0, 234646))
# random.shuffle(shuffled_indices)
# with open(os.path.join(base_maths, "shuffled_indices.json"), "w") as file:
#     json.dump(shuffled_indices, file)
#
# rename_and_move_images()
# produce_random_ordering_and_labels()



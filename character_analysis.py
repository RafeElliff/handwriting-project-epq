import segment_scans

import tensorflow_datasets

import numpy

training_dataset, dataset_info = tensorflow_datasets.load(
    'emnist/byclass',
    split='train[:10000]',
    #Loading all of the data takes a fair bit of time. I'm choosing to only load the first 10,000 for now and when I get to proper training obviously I'll load more.
    as_supervised=True,
    with_info=True)

id_to_letters = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "A",
    11: "B",
    12: "C",
    13: "D",
    14: "E",
    15: "F",
    16: "G",
    17: "H",
    18: "I",
    19: "J",
    20: "K",
    21: "L",
    22: "M",
    23: "N",
    24: "O",
    25: "P",
    26: "Q",
    27: "R",
    28: "S",
    29: "T",
    30: "U",
    31: "V",
    32: "W",
    33: "X",
    34: "Y",
    35: "Z",
    36: "a",
    37: "b",
    38: "c",
    39: "d",
    40: "e",
    41: "f",
    42: "g",
    43: "h",
    44: "i",
    45: "j",
    46: "k",
    47: "l",
    48: "m",
    49: "n",
    50: "o",
    51: "p",
    52: "q",
    53: "r",
    54: "s",
    55: "t",
    56: "u",
    57: "v",
    58: "w",
    59: "x",
    60: "y",
    61: "z",
}
print("Image Loading Done.")
images = []
labels = []
counter = 0
for image, label in training_dataset:
    counter = counter + 1
    if counter < 1000:
        image = numpy.array(image)
        transposed = numpy.transpose(image)
        images.append(transposed)
        labels.append(numpy.array(label))

def scale_array_to_0_to_1(numpy_array):
    scaled = numpy.divide(numpy_array, 255)
    return scaled

print(len(images))
print(images[501], scale_array_to_0_to_1(images[501]), id_to_letters[int(labels[501])])


# numpy_array, components, npy_filename = segment_scans.full_segmentation_pipeline("gold standard scan.npy")
# components, npy_arrays_to_analyse, npy_filename = segment_scans.get_npy_images(components, npy_filename, numpy_array)
# for array in npy_arrays_to_analyse:
#     print(array)

# class CONV_layer:
#
#
# class ReLU_layer:
#
# class Linear_Layer:
#
# class POOL_Layer:

# KGAT_0b07f5dd603ee0e590386993d742a798

import numpy as np

import segment_scans

import tensorflow_datasets

import numpy
import math

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




class Linear_Layer:
    def __init__(self, num_of_inputs, num_of_neurons):
        self.num_of_inputs = num_of_inputs
        self.num_of_neurons = num_of_neurons
        weights = numpy.random.randn(num_of_neurons, num_of_inputs) * math.sqrt(2/num_of_inputs)
        self.weights = weights
        self.bias = np.zeros(num_of_neurons)
        self.input = None
    def forward_pass(self, input):
        self.input = input
        output = numpy.matmul(input, self.weights)
        return output
    def backprop(self, dOutput):
        dInput = numpy.matmul(dOutput, numpy.transpose(self.weights))
        dWeights = dOutput * self.input
        dBias = dOutput
        return dInput, dWeights, dBias




class ReLU_Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_pass(self, input):
        self.input = input
        output = numpy.maximum(0, input)
        self.output = output
        return output

    def backprop(self, dOutput): # The naming here can be a bit confusing. 'Input' and 'Output' always refer to the forward pass, so confusingly here we take dOutput and compute dInput from that.
        input = self.input
        output = self.output
        mask = numpy.minimum(1, output)
        dInput = dOutput * mask
        return dInput

def SVM_loss(answers, ground_truth):
    correct_score = answers[ground_truth]
    total_loss = 0
    margin = 1
    violating_classes = 0
    dLoss = numpy.zeros((len(answers), 1)) #dLoss/dAnswers


    for index in range (0, len(answers)):
        if answers[index] - correct_score + margin > 0 and index != ground_truth:
            violating_classes = violating_classes - 1
            dLoss[index] = 1
            total_loss = total_loss + (answers[index] - correct_score + margin)

    dLoss[ground_truth] = violating_classes

    return total_loss, dLoss



import tensorflow_datasets
import numpy
import math
import time
training_dataset, dataset_info = tensorflow_datasets.load(
    'emnist/byclass',
    split='train[:16000]',
    #Processing all of the data takes a fair bit of time. I'm choosing to only load the first few samples for now and when I get to proper training obviously I'll load more.
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
images = []
labels = []
counter = 0
training_images = []
training_labels = []
def scale_array_to_0_to_1(numpy_array):
    scaled = numpy.divide(numpy_array, 255)
    return scaled

for image, label in training_dataset:

    image = numpy.array(image)
    scaled = scale_array_to_0_to_1(image)
    transposed = numpy.transpose(scaled)
    flattened = transposed.flatten()
    training_images.append(flattened)
    training_labels.append(numpy.array(label))

print("Image preparation done")


layer_ids = [-1] #-1 does not correspond to a layer, it is just to avoid error handling


class Linear_Layer:
    def __init__(self, num_of_inputs, num_of_neurons):
        self.num_of_inputs = num_of_inputs
        self.num_of_neurons = num_of_neurons
        weights = numpy.random.randn(num_of_inputs, num_of_neurons) * math.sqrt(2/num_of_inputs)
        self.weights = weights
        self.bias = numpy.zeros(num_of_neurons)
        self.input = None
        self.type = "Linear Layer"
        self.id = layer_ids[-1] + 1
        layer_ids.append(self.id)
    def forward_pass(self, input):
        self.input = input
        output = numpy.matmul(input, self.weights) + self.bias
        return output
    def backprop(self, dOutput):
        dInput = numpy.matmul(dOutput, numpy.transpose(self.weights))
        dWeights = numpy.outer(self.input, dOutput)
        dBias = dOutput
        gradients[self.id]["weights"] = dWeights
        gradients[self.id]["bias"] = dBias
        return dInput


class ReLU_Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.type = "ReLU Layer"
        self.id = layer_ids[-1] + 1
        self.mask = None
        layer_ids.append(self.id)

    def forward_pass(self, input):
        self.input = input
        output = numpy.maximum(0, input)
        self.mask = input > 0
        self.output = output
        return output

    def backprop(self, dOutput): # The naming here can be a bit confusing. 'Input' and 'Output' always refer to the forward pass, so confusingly here we take dOutput and compute dInput from that.
        dInput = dOutput * self.mask
        return dInput

class Adam_optimiser:
    def __init__(self, learning_rate, layers, beta1, beta2, eps):
        self.learning_rate = learning_rate
        self.layers = layers
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.timestep = 0

        self.m = {}
        self.v = {}
    def zero_gradients(self, layers):
        for layer in layers:
            if layer.type == "Linear Layer":

                self.m[layer.id] = {
                    "weights": numpy.zeros_like(layer.weights),
                    "bias": numpy.zeros_like(layer.bias)
                }
                self.v[layer.id] = {
                    "weights": numpy.zeros_like(layer.weights),
                    "bias": numpy.zeros_like(layer.bias)
                }
    def step(self, gradients):
        self.timestep = self.timestep + 1
        for layer in self.layers:
            if layer.type == "Linear Layer":
                m_weights = self.m[layer.id]["weights"]
                m_bias = self.m[layer.id]["bias"]
                v_weights = self.v[layer.id]["weights"]
                v_bias = self.v[layer.id]["bias"]
                dWeights = gradients[layer.id]["weights"]
                dBias = gradients[layer.id]["bias"]

                m_weights = self.beta1 * m_weights+ (1 - self.beta1) * dWeights
                v_weights = self.beta2 * v_weights + (1 - self.beta2) * (dWeights ** 2)
                m_weights_protected = m_weights / (1 - self.beta1 ** self.timestep)
                v_weights_protected = v_weights / (1 - self.beta2 ** self.timestep)
                layer.weights = layer.weights - (self.learning_rate * m_weights_protected) / (numpy.sqrt(v_weights_protected) + self.eps)
                self.m[layer.id]["weights"] = m_weights
                self.v[layer.id]["weights"] = v_weights

                m_bias = self.beta1 * m_bias + (1 - self.beta1) * dBias
                v_bias = self.beta2 * v_bias + (1 - self.beta2) * (dBias ** 2)
                m_bias_protected = m_bias / (1 - self.beta1 ** self.timestep)
                v_bias_protected = v_bias / (1 - self.beta2 ** self.timestep)
                change = (self.learning_rate * m_bias_protected) / (numpy.sqrt(v_bias_protected) + self.eps)
                layer.bias = layer.bias - change
                self.m[layer.id]["bias"] = m_bias
                self.v[layer.id]["bias"] = v_bias


def SVM_loss(answers, ground_truth):
    correct_score = answers[ground_truth]
    total_loss = 0
    margin = 1
    violating_classes = 0
    correct = False
    dLoss = numpy.zeros((len(answers))) #dLoss/dAnswers


    for index in range (0, len(answers)):
        if answers[index] - correct_score + margin > 0 and index != ground_truth:
            violating_classes = violating_classes - 1
            dLoss[index] = 1
            total_loss = total_loss + (answers[index] - correct_score + margin)

    dLoss[ground_truth] = violating_classes

    prediction = numpy.argmax(answers)
    if prediction == ground_truth:
        correct = True

    return total_loss, dLoss, correct


layer_1 = Linear_Layer(784, 256)
layer_2 = ReLU_Layer()
layer_3 = Linear_Layer(256, 128)
layer_4 = ReLU_Layer()
layer_5 = Linear_Layer(128, 64)
layer_6 = ReLU_Layer()
layer_7 = Linear_Layer(64, 62)
layers = [
    layer_1,
    layer_2,
    layer_3,
    layer_4,
    layer_5,
    layer_6,
    layer_7
]
optimiser = Adam_optimiser(0.001, layers, 0.9, 0.999, 0.00000001)
optimiser.zero_gradients(layers)
loss_best = 10 ** 10
epoch_loss_best = -1
for epoch in range (0, 10**10):

    time_tuple = time.localtime()
    time_at_start = str(time_tuple[3]) + ":" + str(time_tuple[4]) + ":" + str(time_tuple[5])
    seconds_at_start = time.time()
    print(f"time at start of epoch {epoch} = {time_at_start}")
    loss_total = 0
    total_correct = 0
    for index in range (0, len(training_images)):
        gradients = {}
        for layer in layers:
            gradients[layer.id] = {
                "weights": None,
                "bias": None
            }
        flattened_image = training_images[index]
        ground_truth = training_labels[index]
        forward = flattened_image
        for layer in layers:
            forward = layer.forward_pass(forward)

        loss, dLoss, correct = SVM_loss(forward, ground_truth)
        loss_total = loss_total + loss

        backward = dLoss
        layers.reverse()
        for layer in layers:
            backward = layer.backprop(backward)
        layers.reverse()
        optimiser.step(gradients)
        if correct:
            total_correct = total_correct + 1
    average_loss = round(loss_total/len(training_images), 2)
    seconds_at_finish = time.time()

    average_correct = total_correct / len(training_images)
    print(f"Epoch {epoch}, loss = {average_loss}, time to run = {round((seconds_at_finish-seconds_at_start), 2)}, accuracy % = {round(average_correct * 100, 5)}")
    if average_loss < loss_best:
        loss_best = average_loss
        epoch_loss_best = epoch

    if epoch - epoch_loss_best > 5:
        print(f"Best epoch = {epoch_loss_best}, with loss {loss_best}")
        break






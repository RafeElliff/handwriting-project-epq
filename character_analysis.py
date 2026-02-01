import os
import random
import tensorflow_datasets
import numpy
import math
import time
import pickle
from helper_functions import scale_array_to_0_to_1
from load_images import get_EMNIST_images, get_maths_images, get_full_set
import json
from numpy.lib.stride_tricks import as_strided
base_training_data = (r"C:\Users\rafee\PycharmProjects\data-for-handwriting-epq")


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






class Linear_Layer:
    def __init__(self, num_of_inputs, num_of_neurons, classifier):
        self.classifier = classifier
        self.num_of_inputs = num_of_inputs
        self.num_of_neurons = num_of_neurons
        weights = numpy.random.randn(num_of_inputs, num_of_neurons) * math.sqrt(2/num_of_inputs)
        self.weights = weights
        self.bias = numpy.zeros(num_of_neurons)
        self.input = None
        self.type = "Linear Layer"
        self.id = classifier.layer_ids[-1] + 1
        classifier.layer_ids.append(self.id)
    def forward_pass(self, input):
        self.input = input
        output = numpy.matmul(input, self.weights) + self.bias
        return output
    def backprop(self, dOutput):
        dInput = numpy.matmul(dOutput, numpy.transpose(self.weights))
        dWeights = numpy.matmul(numpy.transpose(self.input), dOutput) / dOutput.shape[0]
        dBias = numpy.sum(dOutput, axis=0)
        self.classifier.gradients[self.id]["weights"] = dWeights
        self.classifier.gradients[self.id]["bias"] = dBias
        return dInput


class ReLU_Layer:
    def __init__(self, classifier):
        self.input = None
        self.output = None
        self.type = "ReLU Layer"
        self.id = classifier.layer_ids[-1] + 1
        self.mask = None
        self.classifier = classifier
        classifier.layer_ids.append(self.id)

    def forward_pass(self, input):
        self.input = input
        output = numpy.maximum(0, input)
        self.mask = input > 0
        self.output = output
        return output

    def backprop(self, dOutput): # The naming here can be a bit confusing. 'Input' and 'Output' always refer to the forward pass, so confusingly here we take dOutput and compute dInput from that.
        dInput = dOutput * self.mask
        return dInput


class CONV_Layer:
    def __init__(self, kernel_size, num_of_filters, stride, input_depth, input_width, classifier):
        self.kernel_size = kernel_size
        self.classifier = classifier
        self.num_of_filters = num_of_filters
        self.stride = stride
        self.padding = (kernel_size-1)//2
        self.input_depth = input_depth
        self.input_width = input_width
        self.id = classifier.layer_ids[-1] + 1
        classifier.layer_ids.append(self.id)
        self.type = "CONV_Layer"
        self.initialise_filter_weights()
    def initialise_filter_weights(self):
        self.filters = {}
        num_of_filters = self.num_of_filters
        for filter in range (0, num_of_filters):
            weights = numpy.random.randn(self.kernel_size, self.kernel_size, self.input_depth) * math.sqrt(2/self.num_of_filters)
            bias = 0
            self.filters[filter] = {
                "weights": weights,
                "bias": bias
            }

    def im2col(self, images_batch):
        batch_size, height, width, depth = images_batch.shape

        output_height = (height - self.kernel_size) // self.stride + 1
        output_width = (width - self.kernel_size) // self.stride + 1
        stride_0, stride_1, stride_2, stride_3 = images_batch.strides
        strided = as_strided(
            images_batch,
            shape = (batch_size, output_height, output_width, self.kernel_size, self.kernel_size, depth),
            strides=(stride_0, stride_1 * self.stride, stride_2 * self.stride, stride_1, stride_2, stride_3),
            writeable = False
        )
        cols = strided.reshape(batch_size * output_height * output_width, -1)
        transposed = cols.transpose()
        return transposed

    def col2im(self, four_d_matrix):
        padded_forward_pass = self.input
        original_image = self.original_images
        dInput_padded = numpy.zeros_like(padded_forward_pass)
        batch_size, height_orig, width_orig, depth_orig = original_image.shape

        col_reshaped = four_d_matrix.reshape(
            self.kernel_size, self.kernel_size, depth_orig,
            batch_size, height_orig, width_orig
        )

        for kernel_height in range(0, self.kernel_size):
            for kernel_width in range(0, self.kernel_size):
                height_start = kernel_height
                height_end = height_start + height_orig * self.stride
                width_start = kernel_width
                width_end = width_start + width_orig * self.stride

                dInput_padded[:, height_start:height_end:self.stride, width_start:width_end:self.stride, :] += col_reshaped[kernel_height, kernel_width, :, :, :, :].transpose(1, 2, 3, 0)

        dInput = dInput_padded[:, self.padding:-self.padding, self.padding:-self.padding, :]
        return dInput

    def full_weights_matrix(self):
        full_weights_list = []
        full_bias_list = []
        for id, inner_dictionary in self.filters.items():
            weight_for_id = inner_dictionary["weights"]
            bias_for_id = inner_dictionary["bias"]
            flattened_weights = weight_for_id.flatten()
            full_weights_list.append(flattened_weights)
            full_bias_list.append(bias_for_id)
        weights_numpy = numpy.array(full_weights_list)
        bias_vector = numpy.array(full_bias_list)
        bias_matrix = bias_vector.reshape(self.num_of_filters, 1)
        return weights_numpy, bias_matrix

    def forward_pass(self, images_batch):
        time_before_total = time.time()
        batch_size, orig_height, orig_width, orig_depth = images_batch.shape
        self.batch_size = batch_size
        padded = numpy.pad(images_batch, ((0, 0),(self.padding, self.padding), (self.padding, self.padding), (0,0)), mode='constant')
        self.input = padded
        self.original_images = images_batch
        batch_size, orig_height, orig_width, orig_depth= images_batch.shape
        time_before_im2col= time.time()
        patch_matrix = (self.im2col(padded))
        time_after_im2col = time.time()
        weights_matrix, bias_matrix = self.full_weights_matrix()
        self.weights = weights_matrix
        results = numpy.matmul(weights_matrix, patch_matrix) + bias_matrix
        reshaped_results = results.reshape(self.num_of_filters, batch_size, orig_height, orig_width,)
        transposed_results = numpy.transpose(reshaped_results, (1, 2, 3, 0))

        self.patch_matrix = patch_matrix
        time_after_total = time.time()
        # print("total time forward", self.id, time_after_total - time_before_total)
        # print("im2col time forward", self.id, time_after_im2col - time_before_im2col)
        return transposed_results

    def backprop(self, dOutput):
        time_before = time.time()
        weights = self.weights
        batch_size, height, width, depth = dOutput.shape
        dBias = []
        for id in range(0, self.num_of_filters):
            dBias.append(numpy.sum(dOutput[:, :, :, id], axis = (0,1,2)))
        dBias = numpy.array(dBias)
        dOutput_Transposed = numpy.transpose(dOutput, (3, 0, 1, 2))
        dOutput_reshaped = numpy.reshape(dOutput_Transposed, (self.num_of_filters, -1))
        patch_matrix = self.patch_matrix
        transposed_patch_matrix = patch_matrix.transpose()
        dWeights = numpy.matmul(dOutput_reshaped, transposed_patch_matrix) / batch_size
        input_to_col2im_reshaped = numpy.matmul(numpy.transpose(weights), dOutput_reshaped)
        total_patches = height * width * batch_size
        input_to_col2im = input_to_col2im_reshaped.reshape(self.kernel_size, self.kernel_size, self.input_depth, total_patches)
        self.classifier.gradients[self.id] = {
            "weights": dWeights,
            "bias": dBias
        }
        time_before_col2im = time.time()
        dInput = self.col2im(input_to_col2im)
        time_after = time.time()
        # print("total time backward", self.id, time_after - time_before)
        # print("col2im time backward", self.id, time_after - time_before_col2im)
        return dInput



class Flatten_Layer:
    def __init__(self, classifier):
        self.id = classifier.layer_ids[-1] + 1
        classifier.layer_ids.append(self.id)
        self.type = "Flatten_Layer"
    def forward_pass(self, input):
        self.input_shape = input.shape
        flattened = input.reshape(input.shape[0], -1)
        return flattened

    def backprop(self, dOutput_flat):
        dOutput_3d = numpy.reshape(dOutput_flat, self.input_shape)
        return dOutput_3d

def get_random_hyperparams():
    LR_LB = 0.0003 #lower bound/upper bound
    LR_UB = 0.003
    default_LR = 0.001
    LR_range_test = True #Set this to true if you want to be checking the LR range.
    batch_size_options = [65, 130]
    default_batch_size = 130
    batch_size_test = True
    L2_LB = 0.01
    L2_UB = 0.1
    default_L2 = 0.01
    L2_range_test = True
    possible_filter_sizes = [(16, 32, 64), (32, 48, 64), (32, 64, 128)]
    default_filter_sizes = (32, 64, 128)
    filter_size_test = True
    #Here, the first number per tuple is the type of decay. 0 = no decay, 1 = linear, 2 = exponential. The second number is the rate (where applicable), and the third is the step size (where applicable)
    learning_rate_decay_options = [(0, 0, 0), (1, 0.95, 5), (1, 0.9, 5), (1, 0.8, 5), (2, 0.96, 0), (2, 0.93, 0), (2, 0.9, 0) ]
    LR_decay_test = False
    LR_decay_default = learning_rate_decay_options[0]
    random_LR = random.uniform(LR_LB, LR_UB)
    random_L2 = random.uniform(L2_LB, L2_UB)
    random_batch_size = random.choice(batch_size_options)
    random_filter_size = random.choice(possible_filter_sizes)
    random_LR_decay = random.choice(learning_rate_decay_options)

    if L2_range_test:
        final_L2 = random_L2
    else:
        final_L2 = default_L2
    if LR_range_test:
        final_LR = random_LR
    else:
        final_LR = default_LR
    if batch_size_test:
        final_batch_size = random_batch_size
    else:
        final_batch_size = default_batch_size
    if filter_size_test:
        final_filter_size = random_filter_size
    else:
        final_filter_size = default_filter_sizes
    if LR_decay_test:
        final_LR_decay = random_LR_decay
    else:
        final_LR_decay = LR_decay_default

    return final_LR, final_batch_size, final_L2, final_filter_size, final_LR_decay


class Adam_Optimiser:
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
            elif layer.type == "CONV_Layer":
                weights_matrix, bias_matrix = layer.full_weights_matrix()
                bias_vector = bias_matrix.flatten()
                self.m[layer.id] = {
                    "weights": numpy.zeros_like(weights_matrix),
                    "bias": numpy.zeros_like(bias_vector)
                }
                self.v[layer.id] = {
                    "weights": numpy.zeros_like(weights_matrix),
                    "bias": numpy.zeros_like(bias_vector)
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

            elif layer.type == "CONV_Layer":
                dWeights = gradients[layer.id]["weights"]
                dBias = gradients[layer.id]["bias"]

                m_weights = self.m[layer.id]["weights"]
                m_bias = self.m[layer.id]["bias"]
                v_weights = self.v[layer.id]["weights"]
                v_bias = self.v[layer.id]["bias"]

                m_weights = self.beta1 * m_weights + (1 - self.beta1) * dWeights
                v_weights = self.beta2 * v_weights + (1 - self.beta2) * (dWeights ** 2)
                m_weights_protected = m_weights / (1 - self.beta1 ** self.timestep)
                v_weights_protected = v_weights / (1 - self.beta2 ** self.timestep)

                weight_updates = - (self.learning_rate * m_weights_protected) / (numpy.sqrt(v_weights_protected) + self.eps)

                for filter_id in range(0, layer.num_of_filters):
                    weight_update_for_filter = weight_updates[filter_id].reshape(layer.kernel_size, layer.kernel_size, layer.input_depth)
                    layer.filters[filter_id]["weights"] = layer.filters[filter_id]["weights"] + weight_update_for_filter

                self.m[layer.id]["weights"] = m_weights
                self.v[layer.id]["weights"] = v_weights

                m_bias = self.beta1 * m_bias + (1 - self.beta1) * dBias
                v_bias = self.beta2 * v_bias + (1 - self.beta2) * (dBias ** 2)
                m_bias_protected = m_bias / (1 - self.beta1 ** self.timestep)
                v_bias_protected = v_bias / (1 - self.beta2 ** self.timestep)

                bias_updates = - (self.learning_rate * m_bias_protected) / (numpy.sqrt(v_bias_protected) + self.eps)

                for filter_id in range(0, layer.num_of_filters):
                    layer.filters[filter_id]["bias"] = layer.filters[filter_id]["bias"] + bias_updates[filter_id]

                self.m[layer.id]["bias"] = m_bias
                self.v[layer.id]["bias"] = v_bias
def SVM_loss_single_image(answers, ground_truth):
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

def batched_SVM(answers_batch, ground_truths_batch): #Answers has shape (batch_size, 62). ground_truth has shape (batch_size)
    total_loss = 0
    all_dLoss = []
    total_correct = 0
    batch_size = len(answers_batch)
    for image in range (0, batch_size):
        answers = answers_batch[image]
        ground_truth = ground_truths_batch[image]
        loss, dLoss, correct = SVM_loss_single_image(answers, ground_truth)
        total_loss = total_loss + loss
        all_dLoss.append(dLoss)
        if correct:
            total_correct = total_correct + 1

    average_loss = total_loss/batch_size
    all_dLoss_matrix = numpy.stack(all_dLoss)
    return average_loss, all_dLoss_matrix, total_correct


def LR_decay(LR_decay_hyperparam, initial_LR, epoch):
    type_of_decay = LR_decay_hyperparam[0]
    rate_of_decay = LR_decay_hyperparam[1]
    step_size_of_decay = LR_decay_hyperparam[2]
    if type_of_decay == 0: #No LR
        return initial_LR
    if type_of_decay == 1: #Linear Decay
        return initial_LR * rate_of_decay ** (epoch/step_size_of_decay)
    if type_of_decay == 2: #Exponential Decay
        return initial_LR * math.e ** -(rate_of_decay * epoch)

def write_new_line_to_file(filename, line):
    filepath_source = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\training data runs"
    filepath = os.path.join(filepath_source, filename)
    with open(filepath, "a") as file:
        file.write(line + "\n")
class Classification_Model_NEW():
    def __init__(self, hyperparams):
        print(hyperparams)
        # write_new_line_to_file("night of 28-1", f"LR = {hyperparams[0]}, batch_size = {hyperparams[1]}, L2_strength = {hyperparams[2]}, filter_sizes = {hyperparams[3]}, LR decay = {hyperparams[4]}")
        self.hyperparams = hyperparams
        self.layer_ids = [-1]  # -1 does not correspond to a layer, it is just to avoid error handling
        layer_0 = CONV_Layer(3, hyperparams[3][0], 1, 1, 28, self)
        layer_1 = ReLU_Layer(self)
        layer_2 = CONV_Layer(3, hyperparams[3][1], 1, hyperparams[3][0], 28, self)
        layer_3 = ReLU_Layer(self)  # dOutput = 28*28*64
        layer_4 = CONV_Layer(3, hyperparams[3][2], 1, hyperparams[3][1], 28, self)  # dOutput = 28*28*128
        layer_5 = Flatten_Layer(self)  # dOutput = 100352
        layer_6 = Linear_Layer(28 * 28 * hyperparams[3][2], 84, self)  # dOutput = 62

        layers = [
            layer_0,
            layer_1,
            layer_2,
            layer_3,
            layer_4,
            layer_5,
            layer_6,
        ]
        self.layers = layers
        self.L2_lambda = hyperparams[2]
        self.optimiser = Adam_Optimiser(hyperparams[0], layers, 0.9, 0.999, 0.00000001)
        self.optimiser.zero_gradients(layers)
        self.best_accuracy = 0
        self.epoch_with_best_accuracy = -1
        self.gradients = {}

    def train(self):
        self.optimiser = Adam_Optimiser(self.hyperparams[0], self.layers, 0.9, 0.999, 0.00000001)
        self.optimiser.zero_gradients(self.layers)
        self.gradients = {}
        for epoch in range(0, 8):
            self.decayed_LR = LR_decay(self.hyperparams[4], self.hyperparams[0], epoch)
            self.optimiser.learning_rate = self.decayed_LR
            time_tuple = time.localtime()
            time_at_start = str(time_tuple[3]) + ":" + str(time_tuple[4]) + ":" + str(time_tuple[5])
            print(f"time at start of epoch {epoch} = {time_at_start}")
            total_correct = 0
            total_EMNIST_images = 30000
            total_maths_images = 9000
            total_images = total_EMNIST_images + total_maths_images #39000
            self.batch_size = self.hyperparams[1]
            batch_size = self.batch_size
            # maths_per_batch = int(batch_size *3/13)
            # EMNIST_per_batch = int(batch_size *10/13)
            update_after_n_batches = 1
            loss_total = 0
            little_batch_size = batch_size
            big_batch_size = 100 * little_batch_size #13000
            #For provided parameters, 3 big batches, each with 100 little batches
            for big_batch in range(0, total_images//(little_batch_size*100)):
                maths_per_big_batch = int(big_batch_size * 3 / 13)
                EMNIST_per_big_batch = int(big_batch_size * 10 / 13)
                time_before_big_loading = time.time()
                training_images, training_labels = get_full_set(maths_starting=maths_per_big_batch * big_batch, maths_finishing=maths_per_big_batch * (big_batch+1), EMNIST_starting=EMNIST_per_big_batch * big_batch, EMNIST_finishing=EMNIST_per_big_batch * (big_batch + 1), training_or_testing="training")
                shuffled_indices = (list(range(0, len(training_images))))
                random.shuffle(shuffled_indices)
                loaded_batch_images = training_images[shuffled_indices]
                loaded_batch_labels = training_labels[shuffled_indices]
                time_after_big_loading = time.time()
                time_for_big_loading = time_after_big_loading - time_before_big_loading
                print("big_loading", round(time_for_big_loading, 3))
                for little_batch in range(0, 100):
                    time_before_loading_data = time.time()
                    images_per_batch =loaded_batch_images[little_batch*little_batch_size: (little_batch+1)*little_batch_size]
                    labels_per_batch =loaded_batch_labels[little_batch*little_batch_size: (little_batch+1)*little_batch_size]
                    forward = images_per_batch
                    ground_truth = labels_per_batch
                    time_after_loading_data = time.time()
                    if little_batch % update_after_n_batches == 0:
                        print(f"Epoch {epoch}, Batch Number {little_batch}/{99} of big batch {big_batch}/{total_images//(little_batch_size*100)-1}")
                    time_at_batch_start = time.time()
                    time_before_layer_declaration = time.time()
                    for layer in self.layers:
                        if layer.type == "Linear Layer":
                            self.gradients[layer.id] = {
                                "weights": None,
                                "bias": None
                            }
                        elif layer.type == "CONV_Layer":
                            self.gradients[layer.id] = {
                                "weights": None,
                                "bias": None
                            }
                    time_after_layer_declaration = time.time()
                    time_for_layer_declaration = time_after_layer_declaration - time_before_layer_declaration
                    time_before_forward_pass = time.time()
                    for layer in self.layers:
                        # time_before_layer = time.time()
                        forward = layer.forward_pass(forward)
                        # time_after_layer = time.time()
                        # time_for_layer = time_after_layer-time_before_layer
                        # print("Layer", layer.id, time_for_layer)
                    time_after_forward_pass = time.time()
                    time_for_forward_pass = time_after_forward_pass - time_before_forward_pass
                    time_before_L2 = time.time()
                    loss, dLoss, correct = batched_SVM(forward, ground_truth)
                    L2_loss = 0
                    for layer in self.layers:
                        if layer.type == "Linear Layer":
                            L2_loss = L2_loss + numpy.sum(layer.weights ** 2)
                        elif layer.type == "CONV_Layer":
                            for filter_id in layer.filters:
                                weights = layer.filters[filter_id]["weights"]
                                L2_loss = L2_loss + numpy.sum(weights ** 2)
                    L2_loss = (self.L2_lambda / 2) * L2_loss
                    time_after_L2 = time.time()
                    time_for_L2 = time_after_L2 - time_before_L2
                    loss_total = loss_total + loss + L2_loss
                    total_correct = total_correct + correct
                    backward = dLoss
                    time_before_backprop = time.time()
                    for layer in reversed(self.layers):
                        # time_before_layer = time.time()
                        backward = layer.backprop(backward)
                        # time_after_layer = time.time()
                        # time_for_layer = time_after_layer - time_before_layer
                        # print("Layer", layer.id, time_for_layer)


                    time_after_backprop = time.time()
                    time_for_backprop = time_after_backprop - time_before_backprop
                    time_before_optimiser_step = time.time()
                    for layer in self.layers:
                        if layer.type == "Linear Layer":
                            self.gradients[layer.id]["weights"] += self.L2_lambda * layer.weights
                        elif layer.type == "CONV_Layer":
                            weights_matrix, bias_matrix = layer.full_weights_matrix()
                            self.gradients[layer.id]["weights"] += self.L2_lambda * weights_matrix
                    time_after_optimiser_step = time.time()
                    time_for_optimiser_step = time_after_optimiser_step-time_before_optimiser_step
                    time_at_batch_end = time.time()
                    if little_batch % update_after_n_batches == 0:
                        print(f"Time for Batch = {round(time_at_batch_end - time_at_batch_start, 5)}, time per image = {round(((time_at_batch_end - time_at_batch_start)/batch_size), 5)}")
                        print("data_loading", round(time_after_loading_data-time_before_loading_data, 3))
                        print("layer_declaration", round(time_for_layer_declaration, 3))
                        print("forward_pass", round(time_for_forward_pass, 3))
                        print("L2", round(time_for_L2, 3))
                        print("backprop", round(time_for_backprop, 3))
                        print("optimiser_step", round(time_for_optimiser_step, 3))

        return self.accuracy_check()

    def save_parameters(self):
        for layer in self.layers:
            filename = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\trained_model_data\layer_" + str(layer.id)
            with open(filename, "wb") as file:
                pickle.dump(layer, file)

    def load_parameters(self):
        layers_with_params = []
        for layer in self.layers:
            filename = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\trained_model_data\layer_" + str(layer.id)
            with open(filename, "rb") as file:
                layer = pickle.load(file)
                layers_with_params.append(layer)
        self.layers = layers_with_params
        return layers_with_params

    def accuracy_check(self):
        time_before_accuracy_check = time.time()
        correct_counter = 0
        #Onle one of the following lines should be uncommented. The top one uses the testing set, the bottom one uses the validation set.
        #testing_images, testing_labels = get_full_set(0, 24000, 0, 80000, "testing")
        testing_images, testing_labels = get_full_set(182000, 208000, 600000, 697000, "training")
        for batch in range (0, 800):
            starting_index = batch*130
            finishing_index = (batch+1)*130
            forward = testing_images[starting_index:finishing_index]
            labels = testing_labels[starting_index:finishing_index]
            for layer in self.layers:
                forward = layer.forward_pass(forward)
            predictions = numpy.argmax(forward, axis=1)
            # print(predictions.shape)
            for prediction_index in range(0, 130):
                prediction = predictions[prediction_index]
                label = labels[prediction_index]
                if prediction == label:
                    correct_counter = correct_counter + 1
        time_after_accuracy_check = time.time()
        time_for_accuracy_check = time_after_accuracy_check-time_before_accuracy_check
        print("Accuracy check", round(time_for_accuracy_check, 3))
        return round((correct_counter / len(testing_images) * 100), 3)

    def get_prediction(self, image):
        forward = image
        for layer in self.layers:
            forward = layer.forward_pass(forward)
        prediction = numpy.argmax(forward)
        certainty = numpy.argmax(forward) #THIS CODE ISN'T FINISHED YET
        return prediction, certainty

# hyperparam_list = []
# for hyperparam_set_number in range(0, 25): #This code is currently commented as it does not need to be run more than once.
#     hyperparam_set = get_random_hyperparams()
#     hyperparam_list.append(hyperparam_set)
#
# with open(os.path.join(base_training_data, "hyperparams.json"), "w") as file:
#     json.dump(hyperparam_list, file)
# with open(os.path.join(base_training_data, "accuracies.json"), "w") as file:
#     json.dump([], file)

def get_progress():
    with open(os.path.join(base_training_data, "hyperparams.json"), "r") as file:
        hyperparams = json.load(file)
    with open(os.path.join(base_training_data, "accuracies.json"), "r") as file:
        accuracies = json.load(file)
    number_of_combinations_ran = len(accuracies)
    remaining_hyperparams = hyperparams[number_of_combinations_ran:]
    for remaining_hyperparam in remaining_hyperparams:
        classifier = Classification_Model_NEW(remaining_hyperparam)
        accuracy = classifier.train()
        # accuracy = classifier.accuracy_check()
        accuracies.append(accuracy)
        with open(os.path.join(base_training_data, "accuracies.json"), "w") as file:
            json.dump(accuracies, file)


get_progress()
##
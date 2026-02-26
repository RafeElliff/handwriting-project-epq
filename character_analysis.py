import os
import random
import tensorflow_datasets
import numpy
import math
import time
import pickle
from helper_functions import scale_array_to_0_to_1, view_numpy_as_png, get_similar_letters, get_percentages_from_forward_pass
from load_images import get_EMNIST_images, get_maths_images, get_full_set
from confirm_which_char import get_percentages_from_forward_pass, get_letter_possibilites, get_user_input
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

#README: it will be difficult to understand what is going on here if you are not already familiar with the project or other, similar projects
#I recommend to read the explanation section of the writeup, or the log entries from 20/10/25 to 21/10/25 (Or you could go straight to the source and read the Stanford course notes page)to get an intuition for how it works

class Linear_Layer:

    def __init__(self, num_of_inputs, num_of_neurons, classifier):
        self.classifier = classifier
        self.num_of_inputs = num_of_inputs
        self.num_of_neurons = num_of_neurons
        weights = numpy.random.randn(num_of_inputs, num_of_neurons).astype(numpy.float32) * math.sqrt(2 / num_of_inputs)
        self.weights = weights
        self.bias = numpy.zeros(num_of_neurons, dtype=numpy.float32) #Standard initialisation of weights and bias vectors
        self.input = None
        self.type = "Linear Layer"
        self.id = classifier.layer_ids[-1] + 1
        classifier.layer_ids.append(self.id)

    def forward_pass(self, input): #Forward pass for linear layer is super simple: just matrix multiplication
        self.input = input
        output = numpy.matmul(input, self.weights) + self.bias
        return output

    def backprop(self, dOutput): #Backprop is similarly simple: the formulae are visible below
        #Worth pointing out here that the naming here can be a bit unintuitive  at first. 'Input' and 'Output' always refer to the forward pass, so confusingly here we take dOutput and compute dInput from that.
        dInput = numpy.matmul(dOutput, numpy.transpose(self.weights))
        dWeights = numpy.matmul(numpy.transpose(self.input), dOutput) / dOutput.shape[0]
        dBias = numpy.sum(dOutput, axis=0) / dOutput.shape[0]
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

    def forward_pass(self, input): #ReLU layer is simple: returns x when x>0, returns 0 otherwise
        self.input = input
        output = numpy.maximum(0, input)
        self.mask = input > 0 #Used in backprop.
        self.output = output
        return output

    def backprop(self,
                 dOutput):
        dInput = dOutput * self.mask #dInput = dOutput where input >0, otherwise dInput = 0.
        return dInput

class CONV_Layer:
    #These are definitely the most complex layer type found in the project.
    def __init__(self, kernel_size, num_of_filters, stride, input_depth, input_width, classifier):
        self.kernel_size = kernel_size
        self.classifier = classifier
        self.num_of_filters = num_of_filters
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.input_depth = input_depth
        self.input_width = input_width
        self.id = classifier.layer_ids[-1] + 1
        classifier.layer_ids.append(self.id)
        self.type = "CONV_Layer"
        self.initialise_filter_weights()

    def initialise_filter_weights(self): #Initialises filters with random values
        self.filters = {}
        num_of_filters = self.num_of_filters
        for filter in range(0, num_of_filters):
            weights = numpy.random.randn(self.kernel_size, self.kernel_size, self.input_depth).astype(numpy.float32) * math.sqrt(2 / self.num_of_filters)
            bias = 0
            self.filters[filter] = {
                "weights": weights,
                "bias": bias
            }

    def im2col(self, images_batch): #For more on the im2col and col2im algorithms, and where these implementations came from, look at log entry from 31/1/26
        #The gist of this algorithm is that it converts images into a patch matrix
        batch_size, height, width, depth = images_batch.shape

        output_height = (height - self.kernel_size) // self.stride + 1
        output_width = (width - self.kernel_size) // self.stride + 1
        stride_0, stride_1, stride_2, stride_3 = images_batch.strides #This function returns how many bytes in memory it's necessary to move to access the next position in the given dimension - See log for more detail on the workings of this algorithm
        strided = as_strided(
            images_batch,
            shape=(batch_size, output_height, output_width, self.kernel_size, self.kernel_size, depth),
            strides=(stride_0, stride_1 * self.stride, stride_2 * self.stride, stride_1, stride_2, stride_3),
            writeable=False
        ) #This creates a 'view' of the numpy array with two extra dimensions representing kernel height and width
        # Currently, these have shape (batch_size, height, width, kernel_size, kernel_size, depth)
        cols = strided.reshape(batch_size * output_height * output_width, -1)
        #This reshape is needed to convert each position in the kernel to a patch, as is the purpose of the im2col algorithm
        transposed = cols.transpose()
        #Transpose is needed for matrix multiplication to work
        return transposed

    def col2im(self, four_d_matrix):
        #This is the backprop version of the im2col function: given a matrix of patches, this seeks to reconstruct the original image
        padded_forward_pass = self.input
        original_image = self.original_images
        dInput_padded = numpy.zeros_like(padded_forward_pass, dtype=numpy.float32) #Initialises the matrix so it can be indexed later

        batch_size, height_orig, width_orig, depth_orig = original_image.shape

        col_reshaped = four_d_matrix.reshape(
            self.kernel_size, self.kernel_size, depth_orig,
            batch_size, height_orig, width_orig
        ) #Does the opposite reshape to that which is in im2col, to 'reverse' the algorithm.

        for kernel_height in range(0, self.kernel_size): #It is very difficult to do this operation without any loops. As a result, I wanted to iterate over the smallest possible thing, which here is the pixels in the kernel
            for kernel_width in range(0, self.kernel_size):
                height_start = kernel_height
                height_end = height_start + height_orig * self.stride
                width_start = kernel_width
                width_end = width_start + width_orig * self.stride
                #The use of a += is not intuitive here: in practice, each pixel has contributed to up to 9 patches, so gradients must be accumulated per pixel
                dInput_padded[:, height_start:height_end:self.stride, width_start:width_end:self.stride, :] += col_reshaped[kernel_height, kernel_width, :, :, :, :].transpose(1, 2, 3, 0)

        dInput = dInput_padded[:, self.padding:-self.padding, self.padding:-self.padding, :]
        return dInput

    def full_weights_matrix(self): #Gets the weights matrix and bias matrix from whatever is currently stored in the dictionary (whatever weights and biases were produced from the current optimiser state)
        full_weights_list = []
        full_bias_list = []
        for id, inner_dictionary in self.filters.items():
            weight_for_id = inner_dictionary["weights"]
            bias_for_id = inner_dictionary["bias"]
            flattened_weights = weight_for_id.flatten()
            full_weights_list.append(flattened_weights)
            full_bias_list.append(bias_for_id)
        weights_numpy = numpy.array(full_weights_list, dtype=numpy.float32)
        bias_vector = numpy.array(full_bias_list, dtype=numpy.float32)
        bias_matrix = bias_vector.reshape(self.num_of_filters, 1)
        return weights_numpy, bias_matrix

    def forward_pass(self, images_batch):
        time_before_total = time.time()
        batch_size, orig_height, orig_width, orig_depth = images_batch.shape
        self.batch_size = batch_size
        padded = numpy.pad(images_batch, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                           mode='constant') #Convolution reduces the size of the image. It must, therefore, be padded to 30*30 before applying convolution
        self.input = padded
        self.original_images = images_batch
        #Pads the images and stores the original images for reference
        batch_size, orig_height, orig_width, orig_depth = images_batch.shape
        time_before_im2col = time.time()
        patch_matrix = (self.im2col(padded)) #Uses im2col to get a patch matrix
        time_after_im2col = time.time()
        weights_matrix, bias_matrix = self.full_weights_matrix()
        self.weights = weights_matrix
        results = numpy.matmul(weights_matrix, patch_matrix) + bias_matrix #Computes the result with another matrix multiplication
        reshaped_results = results.reshape(self.num_of_filters, batch_size, orig_height, orig_width, )
        transposed_results = numpy.transpose(reshaped_results, (1, 2, 3, 0)) #Needs to be transposed after the reshape to keep shape consistent

        self.patch_matrix = patch_matrix #Stores patch matrix for backprop
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
            dBias.append(numpy.sum(dOutput[:, :, :, id], axis=(0, 1, 2)))
        dBias = numpy.array(dBias, dtype=numpy.float32) / dOutput.shape[0] #dBias calculation is relatively simple
        dOutput_Transposed = numpy.transpose(dOutput, (3, 0, 1, 2))
        dOutput_reshaped = numpy.reshape(dOutput_Transposed, (self.num_of_filters, -1)) #Reshape and transpose needed for shape consistency
        patch_matrix = self.patch_matrix
        transposed_patch_matrix = patch_matrix.transpose()
        dWeights = numpy.matmul(dOutput_reshaped, transposed_patch_matrix) / batch_size #dWeights calculation is also relatively simple
        #dInput is the hardest one to work with - see col2im function for explanation of this
        input_to_col2im_reshaped = numpy.matmul(numpy.transpose(weights), dOutput_reshaped)
        total_patches = height * width * batch_size
        input_to_col2im = input_to_col2im_reshaped.reshape(self.kernel_size, self.kernel_size, self.input_depth,
                                                           total_patches) #Gets the columns by reversing the forward pass process
        #Stores the gradients in the gradients dictionary
        self.classifier.gradients[self.id] = {
            "weights": dWeights,
            "bias": dBias
        }
        time_before_col2im = time.time()
        dInput = self.col2im(input_to_col2im) #Applies col2im to get dInput
        time_after = time.time()
        #Del statements remove variables from memory. These variables are huge and so removing them can help memory management significantly
        del self.patch_matrix
        del self.input
        del self.original_images
        del self.weights
        # Also delete intermediate calculations:
        del input_to_col2im, input_to_col2im_reshaped
        del dOutput_reshaped, dOutput_Transposed
        return dInput # #These are huge pieces of data and are taking up valuable memory space

class Flatten_Layer:
    #Flatten layers are not true layers in that they do not alter the input numerically, but it is common to include them on the layer stack
    def __init__(self, classifier):
        self.id = classifier.layer_ids[-1] + 1
        classifier.layer_ids.append(self.id)
        self.type = "Flatten_Layer"
    #The purpose of a flatten is the convert the output of a CONV layer into the input of a linear layer. The workings of it are relatively intuitive.
    def forward_pass(self, input):
        self.input_shape = input.shape
        flattened = input.reshape(input.shape[0], -1)
        return flattened

    def backprop(self, dOutput_flat):
        dOutput_3d = numpy.reshape(dOutput_flat, self.input_shape)
        return dOutput_3d

def get_random_hyperparams():
    #This function was used for hyperparameter testing. In the current iteration of the code, the hyperparameters will be constant.
    LR_LB = 0.0002 #lower bound/upper bound
    LR_UB = 0.0005
    default_LR = 0.00035
    LR_range_test = False  #Set this to true if you want to be checking the LR range.
    batch_size_options = [260, 130]
    default_batch_size = 130
    batch_size_test = False
    L2_LB = 0.01
    L2_UB = 0.1
    default_L2 = 0
    L2_range_test = False
    possible_filter_sizes = [(16, 32, 64), (32, 48, 64), (32, 64, 128)]
    default_filter_sizes = (32, 48, 64)
    filter_size_test = False
    #Here, the first number per tuple is the type of decay. 0 = no decay, 1 = linear, 2 = exponential. The second number is the rate (where applicable), and the third is the step size (where applicable)
    learning_rate_decay_options = [(0, 0, 0), (1, 0.95, 5), (1, 0.9, 5), (1, 0.8, 5), (2, 0.96, 0), (2, 0.93, 0),
                                   (2, 0.9, 0)]
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
    #See log for detailed notes on the workings of the Adam optimiser
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
        #This function initialises all m and v values at 0 for all different layers
        for layer in layers:
            if layer.type == "Linear Layer":

                self.m[layer.id] = {
                    "weights": numpy.zeros_like(layer.weights, dtype=numpy.float32),
                    "bias": numpy.zeros_like(layer.bias, dtype=numpy.float32)
                }
                self.v[layer.id] = {
                    "weights": numpy.zeros_like(layer.weights, dtype=numpy.float32),
                    "bias": numpy.zeros_like(layer.bias, dtype=numpy.float32)
                }
            elif layer.type == "CONV_Layer":
                weights_matrix, bias_matrix = layer.full_weights_matrix()
                bias_vector = bias_matrix.flatten()
                self.m[layer.id] = {
                    "weights": numpy.zeros_like(weights_matrix, dtype=numpy.float32),
                    "bias": numpy.zeros_like(bias_vector,dtype=numpy.float32)
                }
                self.v[layer.id] = {
                    "weights": numpy.zeros_like(weights_matrix, dtype=numpy.float32),
                    "bias": numpy.zeros_like(bias_vector, dtype=numpy.float32)
                }

    def step(self, gradients):
        self.timestep = self.timestep + 1
        for layer in self.layers:
            if layer.type == "Linear Layer":
                #Gets the already existing m and v values
                m_weights = self.m[layer.id]["weights"]
                m_bias = self.m[layer.id]["bias"]
                v_weights = self.v[layer.id]["weights"]
                v_bias = self.v[layer.id]["bias"]
                dWeights = gradients[layer.id]["weights"]
                dBias = gradients[layer.id]["bias"]
                #Calculates the new m and v weights values
                m_weights = self.beta1 * m_weights + (1 - self.beta1) * dWeights
                v_weights = self.beta2 * v_weights + (1 - self.beta2) * (dWeights ** 2)
                m_weights_protected = m_weights / (1 - self.beta1 ** self.timestep)
                v_weights_protected = v_weights / (1 - self.beta2 ** self.timestep)
                layer.weights = layer.weights - (self.learning_rate * m_weights_protected) / (
                            numpy.sqrt(v_weights_protected) + self.eps)
                self.m[layer.id]["weights"] = m_weights
                self.v[layer.id]["weights"] = v_weights
                #Calculates the new m and v bias values
                m_bias = self.beta1 * m_bias + (1 - self.beta1) * dBias
                v_bias = self.beta2 * v_bias + (1 - self.beta2) * (dBias ** 2)
                m_bias_protected = m_bias / (1 - self.beta1 ** self.timestep)
                v_bias_protected = v_bias / (1 - self.beta2 ** self.timestep)
                change = (self.learning_rate * m_bias_protected) / (numpy.sqrt(v_bias_protected) + self.eps)
                layer.bias = layer.bias - change
                self.m[layer.id]["bias"] = m_bias
                self.v[layer.id]["bias"] = v_bias

            elif layer.type == "CONV_Layer":
                #Gets existing m and v values
                dWeights = gradients[layer.id]["weights"]
                dBias = gradients[layer.id]["bias"]
                m_weights = self.m[layer.id]["weights"]
                m_bias = self.m[layer.id]["bias"]
                v_weights = self.v[layer.id]["weights"]
                v_bias = self.v[layer.id]["bias"]
                #Works out new m and v values for weights
                m_weights = self.beta1 * m_weights + (1 - self.beta1) * dWeights
                v_weights = self.beta2 * v_weights + (1 - self.beta2) * (dWeights ** 2)
                m_weights_protected = m_weights / (1 - self.beta1 ** self.timestep)
                v_weights_protected = v_weights / (1 - self.beta2 ** self.timestep)

                weight_updates = - (self.learning_rate * m_weights_protected) / (
                            numpy.sqrt(v_weights_protected) + self.eps)

                for filter_id in range(0, layer.num_of_filters):
                    weight_update_for_filter = weight_updates[filter_id].reshape(layer.kernel_size, layer.kernel_size,
                                                                                 layer.input_depth) #Determines what update is needed for each filter
                    layer.filters[filter_id]["weights"] = layer.filters[filter_id]["weights"] + weight_update_for_filter #Applies the necessary update

                self.m[layer.id]["weights"] = m_weights
                self.v[layer.id]["weights"] = v_weights
                #Works out new m and v values for biases
                m_bias = self.beta1 * m_bias + (1 - self.beta1) * dBias
                v_bias = self.beta2 * v_bias + (1 - self.beta2) * (dBias ** 2)
                m_bias_protected = m_bias / (1 - self.beta1 ** self.timestep)
                v_bias_protected = v_bias / (1 - self.beta2 ** self.timestep)

                bias_updates = - (self.learning_rate * m_bias_protected) / (numpy.sqrt(v_bias_protected) + self.eps)

                for filter_id in range(0, layer.num_of_filters):
                    layer.filters[filter_id]["bias"] = layer.filters[filter_id]["bias"] + bias_updates[filter_id]

                self.m[layer.id]["bias"] = m_bias
                self.v[layer.id]["bias"] = v_bias

#These SVM functions are no longer used in the current iteration of the code but were used in the past. They are not used for anything.
def SVM_loss_single_image(answers, ground_truth):
    correct_score = answers[ground_truth]
    total_loss = 0
    margin = 1
    violating_classes = 0
    correct = False
    dLoss = numpy.zeros((len(answers))).astype(numpy.float32)

    for index in range(0, len(answers)):
        if answers[index] - correct_score + margin > 0 and index != ground_truth:
            violating_classes = violating_classes - 1
            dLoss[index] = 1
            total_loss = total_loss + (answers[index] - correct_score + margin)

    dLoss[ground_truth] = violating_classes

    prediction = numpy.argmax(answers)
    if prediction == ground_truth:
        correct = True

    return total_loss, dLoss, correct

def batched_SVM(answers_batch,
                ground_truths_batch):
    total_loss = 0
    all_dLoss = []
    total_correct = 0
    batch_size = len(answers_batch)
    for image in range(0, batch_size):
        answers = answers_batch[image]
        ground_truth = ground_truths_batch[image]
        loss, dLoss, correct = SVM_loss_single_image(answers, ground_truth)
        total_loss = total_loss + loss
        all_dLoss.append(dLoss)
        if correct:
            total_correct = total_correct + 1

    average_loss = total_loss / batch_size
    all_dLoss_matrix = numpy.stack(all_dLoss)
    return average_loss, all_dLoss_matrix, total_correct

def cross_entropy_loss_single_image(answers, ground_truth): #This applies the cross entropy algorithm to a singular image to get a loss
    percentages = get_percentages_from_forward_pass(answers)
    ground_truth_percentage = percentages[ground_truth]
    loss = -numpy.log(ground_truth_percentage + 10**-10) #A very tiny amount is added here because log(0) is undefined, so if percentage = 0 then the program would crash.
    dLoss = percentages.copy()
    dLoss[ground_truth] -= 1

    prediction = numpy.argmax(answers)
    correct = False
    if prediction == ground_truth:
        correct = True

    return loss, dLoss, correct

def batched_cross_entropy(answers_batch,
                ground_truths_batch): #This repeats the cross_entropy_loss_single_image function and aggregates the results to get a loss score and other info from a batch
    total_loss = 0
    all_dLoss = []
    total_correct = 0
    batch_size = len(answers_batch)
    for image in range(0, batch_size):
        answers = answers_batch[image]
        ground_truth = ground_truths_batch[image]
        loss, dLoss, correct = cross_entropy_loss_single_image(answers, ground_truth)
        total_loss = total_loss + loss
        all_dLoss.append(dLoss)
        if correct:
            total_correct = total_correct + 1

    average_loss = total_loss / batch_size
    all_dLoss_matrix = numpy.stack(all_dLoss)
    return average_loss, all_dLoss_matrix, total_correct

def LR_decay(LR_decay_hyperparam, initial_LR, epoch): #Given LR decay info, this applies a decay
    type_of_decay = LR_decay_hyperparam[0]
    rate_of_decay = LR_decay_hyperparam[1]
    step_size_of_decay = LR_decay_hyperparam[2]
    if type_of_decay == 0:  #No LR Decay
        return initial_LR
    if type_of_decay == 1:  #Linear Decay
        return initial_LR * rate_of_decay ** (epoch // step_size_of_decay)
    if type_of_decay == 2:  #Exponential Decay
        return initial_LR * rate_of_decay ** epoch

class Classification_Model:
    def __init__(self, hyperparams):
        print(hyperparams)
        self.hyperparams = hyperparams
        self.layer_ids = [-1]  # -1 does not correspond to a layer, it is just to avoid error handling
        layer_0 = CONV_Layer(3, hyperparams[3][0], 1, 1, 28, self)
        layer_1 = ReLU_Layer(self)
        layer_2 = CONV_Layer(3, hyperparams[3][1], 1, hyperparams[3][0], 28, self)
        layer_3 = ReLU_Layer(self)
        layer_4 = CONV_Layer(3, hyperparams[3][2], 1, hyperparams[3][1], 28, self)
        layer_5 = ReLU_Layer(self)
        layer_6 = Flatten_Layer(self)
        layer_7 = Linear_Layer(28 * 28 * hyperparams[3][2], 84, self)

        layers = [
            layer_0,
            layer_1,
            layer_2,
            layer_3,
            layer_4,
            layer_5,
            layer_6,
            layer_7
        ]
        self.layers = layers # Initialises the layers using the info as defined above
        self.L2_lambda = hyperparams[2]
        self.optimiser = Adam_Optimiser(hyperparams[0], layers, 0.9, 0.999, 1e-8)
        self.optimiser.zero_gradients(layers)
        self.best_accuracy = 0 #This and the below variable are used to determine early stopping
        self.epochs_without_improvement = 0
        self.gradients = {}

    def train(self):
        self.optimiser = Adam_Optimiser(self.hyperparams[0], self.layers, 0.9, 0.999, 1e-8)
        self.optimiser.zero_gradients(self.layers)
        self.gradients = {}
        for epoch in range(0, 35): #The second number is the number of epochs; this should be high enough that plateauing is likely to occur, but not so high that it takes a long time to train.
            self.decayed_LR = LR_decay(self.hyperparams[4], self.hyperparams[0], epoch)
            self.optimiser.learning_rate = self.decayed_LR
            time_tuple = time.localtime()
            time_at_start = str(time_tuple[3]) + ":" + str(time_tuple[4]) + ":" + str(time_tuple[5])
            print(f"time at start of epoch {epoch} = {time_at_start}, lr at epoch = {self.decayed_LR}")
            total_correct = 0
            total_EMNIST_images = 690000
            total_maths_images = 207000
            total_images = total_EMNIST_images + total_maths_images #897000
            self.batch_size = self.hyperparams[1]
            batch_size = self.batch_size
            update_after_n_batches = 25 #After every 25 batches, there is a print statement which shows info about how the process is going such as time taken and loss.
            loss_total = 0
            little_batch_size = batch_size
            big_batch_size = 50 * little_batch_size  #When batch size = 130, this will have value 6500
            #Big batches of images are loaded from which little batches are processed.
            for big_batch in range(0, total_images // (big_batch_size)):
                maths_per_big_batch = int(big_batch_size * 3 / 13)
                EMNIST_per_big_batch = int(big_batch_size * 10 / 13) #Ensures that there are proportional amounts of the two datasets
                time_before_big_loading = time.time()
                training_images, training_labels = get_full_set(maths_starting=maths_per_big_batch * big_batch,
                                                                maths_finishing=maths_per_big_batch * (big_batch + 1),
                                                                EMNIST_starting=EMNIST_per_big_batch * big_batch,
                                                                EMNIST_finishing=EMNIST_per_big_batch * (big_batch + 1),
                                                                training_or_testing="training")
                #Loads the corresponding set of images - these are the images in the 'big batch'.
                shuffled_indices = (list(range(0, len(training_images))))
                random.shuffle(shuffled_indices) #Shuffles the images around
                loaded_batch_images = training_images[shuffled_indices]
                loaded_batch_labels = training_labels[shuffled_indices]
                time_after_big_loading = time.time()
                time_for_big_loading = time_after_big_loading - time_before_big_loading
                print("big_loading", round(time_for_big_loading, 3))
                for little_batch in range(0, 50): #Goes through every individual little batch in the big batch
                    time_before_loading_data = time.time()
                    images_per_batch = loaded_batch_images[
                        little_batch * little_batch_size: (little_batch + 1) * little_batch_size]
                    labels_per_batch = loaded_batch_labels[
                        little_batch * little_batch_size: (little_batch + 1) * little_batch_size] #Calculates what images and labels to pull for a batch and pulls them
                    forward = images_per_batch
                    ground_truth = labels_per_batch
                    time_after_loading_data = time.time()
                    if little_batch % update_after_n_batches == 0:
                        print(
                            f"Epoch {epoch}, Batch Number {little_batch}/{49} of big batch {big_batch}/{total_images // (little_batch_size * 50) - 1}")
                    time_at_batch_start = time.time()
                    time_before_layer_declaration = time.time()
                    for layer in self.layers: #This reinitialises the gradients (dX) for each cycle such that the Adam optimiser works on the correct set of gradients each step
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
                    #These dictionaroes will have values written to them during backprop
                    time_after_layer_declaration = time.time()
                    time_for_layer_declaration = time_after_layer_declaration - time_before_layer_declaration
                    time_before_forward_pass = time.time()
                    for layer in self.layers: #Applies the forward pass
                        forward = layer.forward_pass(forward)
                    time_after_forward_pass = time.time()
                    time_for_forward_pass = time_after_forward_pass - time_before_forward_pass
                    time_before_loss = time.time()
                    loss, dLoss, correct = batched_cross_entropy(forward, ground_truth) #Runs loss calcs
                    L2_loss = 0
                    time_after_loss = time.time()
                    time_for_loss = time_after_loss - time_before_loss
                    time_before_L2 = time.time()
                    #This code calculates the L2 Loss and then adds this to the total loss. This is only for monitoring purposes
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
                    #Goes through the backprop
                    for layer in reversed(self.layers):
                        backward = layer.backprop(backward)

                    time_after_backprop = time.time()
                    time_for_backprop = time_after_backprop - time_before_backprop
                    time_before_optimiser_step = time.time()
                    #This code applies L2 Regularisation to the gradients before calling the optimiser step
                    for layer in self.layers:
                        if layer.type == "Linear Layer":
                            self.gradients[layer.id]["weights"] += self.L2_lambda * layer.weights
                        elif layer.type == "CONV_Layer":
                            weights_matrix, bias_matrix = layer.full_weights_matrix()
                            self.gradients[layer.id]["weights"] += self.L2_lambda * weights_matrix
                    #Optimiser gradients
                    self.optimiser.step(self.gradients)
                    time_after_optimiser_step = time.time()
                    time_for_optimiser_step = time_after_optimiser_step - time_before_optimiser_step
                    time_at_batch_end = time.time()
                    #The following print statements can be uncommented if you want to view the time taken for the CNN to train.
                    # if little_batch % update_after_n_batches == 0:
                    #     print(f"Time for Batch = {round(time_at_batch_end - time_at_batch_start, 5)}, time per image = {round(((time_at_batch_end - time_at_batch_start) / batch_size), 5)}")
                    #     print("data_loading", round(time_after_loading_data - time_before_loading_data, 3))
                    #     print("layer_declaration", round(time_for_layer_declaration, 3))
                    #     print("forward_pass", round(time_for_forward_pass, 3))
                    #     print("loss", round(time_for_loss, 3))
                    #     print("L2", round(time_for_L2, 3))
                    #     print("backprop", round(time_for_backprop, 3))
                    #     print("optimiser_step", round(time_for_optimiser_step, 3))
                    #     print("Loss", loss)
                    del forward, backward, ground_truth #These del statements help with memory management
                    del images_per_batch, labels_per_batch
                    del dLoss, loss
                del loaded_batch_images, loaded_batch_labels
            #This code is responsible for the early stopping algorithm
            accuracies = self.validation_accuracy_check()
            accuracy = accuracies[0]
            if accuracy > self.best_accuracy:
                self.epochs_without_improvement = 0
                self.best_accuracy = accuracy
                self.save_parameters()
            else:
                self.epochs_without_improvement = self.epochs_without_improvement + 1
            if self.epochs_without_improvement > 3:
                return self.best_accuracy
        return self.best_accuracy

    def save_parameters(self): #This function saves the weights and biases / filters to a file such that you can get a trained model without retraining every time you close the project
        for layer in self.layers:
            filename = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\trained_model_data\layer_" + str(layer.id)
            if layer.type == "Linear Layer":
                data_to_save = {
                    "type": layer.type,
                    "weights": layer.weights,
                    "bias": layer.bias
                }
            elif layer.type == "CONV_Layer":
                data_to_save = {
                    "type": layer.type,
                    "filters": layer.filters
                }
            else:
                data_to_save = {"type": layer.type}

            with open (filename, "wb") as file:
                pickle.dump(data_to_save, file)

    def load_parameters(self): #This function loads the weights and biases /filters from the relevant numpy files, such that  predictions can be made from a pretrained model
        for layer in self.layers:
            filename = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\trained_model_data\layer_" + str(layer.id)
            with open(filename, "rb") as file:
                saved_data = pickle.load(file)
            if layer.type == "Linear Layer":
                layer.weights = saved_data["weights"]
                layer.bias = saved_data["bias"]
            elif layer.type == "CONV_Layer":
                layer.filters = saved_data["filters"]

    def testing_accuracy_check(self): #This function does an accuracy check on the training data
        time_before_accuracy_check = time.time()
        correct_counter = 0
        confusable_correct_counter = 0 #The 'confusable' case is when the predictor returns a letter which is not correct but visually almost identical such as 1 and l or O and 0
        accuracies = {}
        for id, character in numbers_to_labels.items():
            accuracies[character] = [0, 0, 0.0] #These three are the total appearances of a character, the total times it is correct, and the corresponding percentage accuracy
        total_images = 97500
        maths_per_big_batch = 1500
        EMNIST_per_big_batch = 5000
        big_batch_size = 6500
        print((total_images//big_batch_size))
        #The first part of this code is not dissimilar to the forward pass in the training function
        for big_batch_start in range(0, (total_images//big_batch_size)):
            testing_images, testing_labels = get_full_set(3000+big_batch_start*maths_per_big_batch, 3000+(big_batch_start+1)*maths_per_big_batch , 10000+big_batch_start*EMNIST_per_big_batch, 10000+(big_batch_start+1)*EMNIST_per_big_batch, "testing")
            print(f"loading big batch {big_batch_start} finished")
            for batch in range(0, len(testing_labels) // 130):
                starting_index = batch * 130
                finishing_index = (batch + 1) * 130
                forward = testing_images[starting_index:finishing_index]
                labels = testing_labels[starting_index:finishing_index]
                for layer in self.layers:
                    forward = layer.forward_pass(forward)
                predictions = numpy.argmax(forward, axis=1)
                for prediction_index in range(0, 130):
                    prediction = predictions[prediction_index]
                    label = labels[prediction_index]
                    label_as_chr = numbers_to_labels[label]
                    old_values = accuracies[label_as_chr]
                    total_appearances = old_values[0]
                    total_correct = old_values[1]
                    if prediction == label: #Prediction is right
                        correct_counter = correct_counter + 1 #Updates accuracies list accordingly
                        new_total_correct_per_char = total_correct + 1
                        new_total_appearances_per_char = total_appearances + 1
                        new_accuracy_per_char = 100 * round(new_total_correct_per_char / new_total_appearances_per_char, 3)
                        new_values = [new_total_appearances_per_char, new_total_correct_per_char, new_accuracy_per_char]
                        accuracies[label_as_chr] = new_values
                    else: #Prediciton is wrong
                        new_total_correct_per_char = total_correct #Updates accuracies list accordingly
                        new_total_appearances_per_char = total_appearances + 1
                        new_accuracy_per_char = 100 * round(new_total_correct_per_char / new_total_appearances_per_char,
                                                            3)
                        new_values = [new_total_appearances_per_char, new_total_correct_per_char, new_accuracy_per_char]
                        accuracies[label_as_chr] = new_values
                    prediction_as_letter = numbers_to_labels[prediction]
                    label_as_letter = numbers_to_labels[label]
                    possible_confusables = get_similar_letters(prediction_as_letter)
                    if label_as_letter in possible_confusables: #Records how often predictions were correct not including confusables
                        confusable_correct_counter = confusable_correct_counter+1
            del testing_images, testing_labels
        time_after_accuracy_check = time.time()
        time_for_accuracy_check = time_after_accuracy_check - time_before_accuracy_check
        print("Accuracy check", round(time_for_accuracy_check, 3))
        print(accuracies)
        return round((correct_counter / total_images * 100), 3), round((confusable_correct_counter / total_images * 100), 3)


    def validation_accuracy_check(self): #This code is almost identical to the code above with the exception that it does not produce the list that gives information about per-character accuracies
        time_before_accuracy_check = time.time()
        correct_counter = 0
        confusable_correct_counter = 0

        total_images = 13000
        maths_per_big_batch = 1500
        EMNIST_per_big_batch = 5000
        big_batch_size = 6500
        print((total_images//big_batch_size))
        for big_batch_start in range(0, (total_images//big_batch_size)):
            testing_images, testing_labels = get_full_set(big_batch_start*maths_per_big_batch, (big_batch_start+1)*maths_per_big_batch , big_batch_start*EMNIST_per_big_batch, (big_batch_start+1)*EMNIST_per_big_batch, "testing")
            print(f"loading big batch {big_batch_start} finished")
            for batch in range(0, len(testing_labels) // 130):
                starting_index = batch * 130
                finishing_index = (batch + 1) * 130
                forward = testing_images[starting_index:finishing_index]
                labels = testing_labels[starting_index:finishing_index]
                for layer in self.layers:
                    forward = layer.forward_pass(forward)
                predictions = numpy.argmax(forward, axis=1)
                for prediction_index in range(0, 130):
                    prediction = predictions[prediction_index]
                    label = labels[prediction_index]
                    if prediction == label:
                        correct_counter = correct_counter + 1

                    prediction_as_letter = numbers_to_labels[prediction]
                    label_as_letter = numbers_to_labels[label]
                    possible_confusables = get_similar_letters(prediction_as_letter)
                    if label_as_letter in possible_confusables:
                        confusable_correct_counter = confusable_correct_counter+1
            del testing_images, testing_labels
        time_after_accuracy_check = time.time()
        time_for_accuracy_check = time_after_accuracy_check - time_before_accuracy_check
        print("Accuracy check", round(time_for_accuracy_check, 3))
        return round((correct_counter / total_images * 100), 3), round((confusable_correct_counter / total_images * 100), 3)

    def get_prediction(self, image): #This is the full classification pipeline for a single image
        image = image.reshape(1, 28, 28, 1)
        forward = image
        for layer in self.layers:
            forward = layer.forward_pass(forward)
        forward_vector = forward[0] #Gets forward pass
        percentages = get_percentages_from_forward_pass(forward_vector) # Gets percentages
        letter_possibilites, percentages = get_letter_possibilites(forward_vector,percentages)
        final_output = get_user_input(image, letter_possibilites, percentage_chances=percentages) #Consults the user if applicable
        return final_output

def get_progress(): #This function was written so that it was not necessary to run all hyperparam combos in a single execution, rather you could run a combination, then restart the code later and keep the progress.
    with open(os.path.join(base_training_data, "hyperparams.json"), "r") as file:
        hyperparams = json.load(file)
    with open(os.path.join(base_training_data, "accuracies.json"), "r") as file:
        accuracies = json.load(file)
    number_of_combinations_ran = len(accuracies)
    remaining_hyperparams = hyperparams[number_of_combinations_ran:] #Works out which hyperparams still to run
    for remaining_hyperparam in remaining_hyperparams: #Runs them
        classifier = Classification_Model(remaining_hyperparam)
        accuracy = classifier.train()
        accuracies.append(accuracy)
        with open(os.path.join(base_training_data, "accuracies.json"), "w") as file:
            json.dump(accuracies, file)

def full_classification_pipeline(list_of_npy_arrays):
    #Given a list of numpy arrays representing files, it returns their corresponding letters including the user interaction
    hyperparam_set = get_random_hyperparams()
    classifier = Classification_Model(hyperparam_set)
    classifier.load_parameters()
    predictions = []
    for index in range(0, len(list_of_npy_arrays)):
        array = list_of_npy_arrays[index]
        prediction = classifier.get_prediction(array)
        predictions.append(prediction)
    return predictions

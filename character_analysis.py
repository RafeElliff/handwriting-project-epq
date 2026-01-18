
import tensorflow_datasets
import numpy
import math
import time
import pickle


datasets, dataset_info = tensorflow_datasets.load(
    "emnist/byclass",
    split=["train[:5]", "test[:1000]"],
    #Processing all of the data takes a fair bit of time. I'm choosing to only load the first few samples for now and when I get to proper training obviously I'll load more.
    as_supervised=True,
    with_info=True)
training_dataset = datasets[0]
testing_dataset = datasets[1]
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
testing_images = []
testing_labels = []
def scale_array_to_0_to_1(numpy_array):
    scaled = numpy.divide(numpy_array, 255)
    scaled_reshaped = numpy.reshape(scaled, (28, 28))
    return scaled_reshaped

for image, label in training_dataset:

    image = numpy.array(image)
    scaled = scale_array_to_0_to_1(image)
    transposed = numpy.transpose(scaled)
    reshaped = numpy.reshape(transposed, (28, 28, 1))
    # flattened = transposed.flatten()
    training_images.append(reshaped)
    training_labels.append(numpy.array(label))

for image, label in testing_dataset:
    image = numpy.array(image)
    scaled = scale_array_to_0_to_1(image)
    transposed = numpy.transpose(scaled)
    reshaped = numpy.reshape(transposed, (28, 28, 1))
    # flattened = transposed.flatten()
    testing_images.append(reshaped)
    testing_labels.append(numpy.array(label))


print("Image preparation done")


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
        classifier.gradients[self.id]["weights"] = dWeights
        classifier.gradients[self.id]["bias"] = dBias
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


class CONV_Layer:
    def __init__(self, kernel_size, num_of_filters, stride, input_depth, input_width):
        self.kernel_size = kernel_size
        self.num_of_filters = num_of_filters
        self.stride = stride
        self.padding = (kernel_size-1)//2
        self.input_depth = input_depth
        self.input_width = input_width
        self.id = layer_ids[-1] + 1
        layer_ids.append(self.id)
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

    def im2col(self, image): # get all_patches. image should be padded
        height, width, depth = image.shape
        num_patches = height - self.padding * 2

        output_patches = []
        for row in range (0, num_patches):
            for column in range (0, num_patches):
                patch = image[row: row+self.kernel_size, column: column+self.kernel_size, :].flatten() #each output here has size 3*3*input_depth
                output_patches.append(patch)

        numpy_array = numpy.array(output_patches)
        transposed = numpy_array.transpose() #For matrix multiplication, it is important that it is 9*input_depth,784, not 784, 9*input_depth as the for loop makes it.

        return transposed

    def col2im(self, four_d_matrix):
        padded_forward_pass = self.input
        original_image = self.original_image
        dInput_padded = numpy.zeros_like(padded_forward_pass)
        height_orig, width_orig, depth_orig = original_image.shape
        padding_px = self.padding
        for row in range(0, height_orig ):
            for column in range(0, width_orig ):
                position_id = (row * height_orig + column)
                dInput_padded[row: row+self.kernel_size, column: column+self.kernel_size, :] += four_d_matrix[:, :, :, position_id] # I dont normally use += as it generally makes code less readable imo but i think it makes it more readable here
        dInput = dInput_padded[self.padding:-self.padding, self.padding:-self.padding, :]

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

    def forward_pass(self, image):
        padded = numpy.pad(image, ((self.padding, self.padding), (self.padding, self.padding), (0,0)), mode='constant')
        self.input = padded
        self.original_image = image
        orig_height, orig_width, orig_depth = image.shape
        patch_matrix = self.im2col(padded)
        weights_matrix, bias_matrix = self.full_weights_matrix()
        self.weights = weights_matrix
        results = numpy.matmul(weights_matrix, patch_matrix) + bias_matrix
        reshaped_results = results.reshape(self.num_of_filters, orig_height, orig_width,)
        transposed_results = numpy.transpose(reshaped_results, (1, 2, 0))

        self.patch_matrix = patch_matrix
        return transposed_results

    def backprop(self, dOutput):
        weights = self.weights
        dBias = []
        for id in range(0, self.num_of_filters):
            dBias.append(numpy.sum(dOutput[:, :, id]))
        dBias = numpy.array(dBias)
        dOutput_Transposed = numpy.transpose(dOutput, (2, 0, 1))
        dOutput_reshaped = numpy.reshape(dOutput_Transposed, (self.num_of_filters, 784))
        patch_matrix = self.patch_matrix
        transposed_patch_matrix = patch_matrix.transpose()
        dWeights = numpy.matmul(dOutput_reshaped, transposed_patch_matrix)
        input_to_col2im_reshaped = numpy.matmul(numpy.transpose(weights), dOutput_reshaped)
        input_to_col2im = numpy.reshape(input_to_col2im_reshaped, (self.kernel_size, self.kernel_size, self.input_depth, 784))
        classifier.gradients[self.id] = {
            "weights": dWeights,
            "bias": dBias
        }
        dInput = self.col2im(input_to_col2im)
        return dInput



class Flatten_Layer:
    def __init__(self):
        self.id = layer_ids[-1] + 1
        layer_ids.append(self.id)
        self.type = "Flatten_Layer"
    def forward_pass(self, input):
        self.input_shape = input.shape
        flattened = input.flatten()
        return flattened

    def backprop(self, dOutput_flat):
        dOutput_3d = numpy.reshape(dOutput_flat, self.input_shape)
        return dOutput_3d

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


layer_ids = [-1] #-1 does not correspond to a layer, it is just to avoid error handling

layer_0 = CONV_Layer(3, 32, 1, 1, 28)
layer_1 = ReLU_Layer()
layer_2 = CONV_Layer(3, 64, 1, 32, 28)
layer_3 = ReLU_Layer()  #dOutput = 28*28*64
layer_4 = CONV_Layer(3, 128, 1, 64, 28)#dOutput = 28*28*128
layer_5 = Flatten_Layer() #dOutput = 100352
layer_6 = Linear_Layer(28*28*128, 62) #dOutput = 62

layers = [
    layer_0,
    layer_1,
    layer_2,
    layer_3,
    layer_4,
    layer_5,
    layer_6,
]

class Classification_Model():
    def __init__(self, layers):
        self.optimiser = Adam_Optimiser(0.001, layers, 0.9, 0.999, 0.00000001)
        self.optimiser.zero_gradients(layers)
        self.loss_best = 10 ** 5
        self.epoch_loss_best = -1
        self.gradients = {}
        self.layers = layers
    def train(self):
        self.optimiser = Adam_Optimiser(0.001, layers, 0.9, 0.999, 0.00000001)
        self.optimiser.zero_gradients(layers)
        self.loss_best = 10 ** 5
        self.epoch_loss_best = -1
        self.gradients = {}
        for epoch in range (0, 25):
            time_tuple = time.localtime()
            time_at_start = str(time_tuple[3]) + ":" + str(time_tuple[4]) + ":" + str(time_tuple[5])
            seconds_at_start = time.time()
            print(f"time at start of epoch {epoch} = {time_at_start}")
            loss_total = 0
            total_correct = 0
            for index in range (0, len(training_images)):
                seconds_at_start_of_image = time.time()
                self.gradients = {}
                for layer in layers:
                    if layer.type == "CONV_Layer":
                        self.gradients[layer.id] = {}
                        for filter in range(0, layer.num_of_filters):
                            self.gradients[layer.id][filter] = {
                                "weights": None,
                                "bias": None
                            }
                    else:
                        self.gradients[layer.id] = {
                            "weights": None,
                            "bias": None
                        }

                initial_image = training_images[index]
                ground_truth = training_labels[index]
                forward = initial_image
                for layer in layers:
                    forward = layer.forward_pass(forward)
                # print(f"Forward for image {index} done")
                seconds_after_forward = time.time()
                # print(f"time for forward = {seconds_after_forward - seconds_at_start_of_image}")
                loss, dLoss, correct = SVM_loss(forward, ground_truth)
                loss_total = loss_total + loss

                backward = dLoss
                layers.reverse()
                for layer in layers:
                    backward = layer.backprop(backward)
                # print(f"Backprop for image {index} done")
                seconds_after_backward = time.time()
                # print(f"time for backward = {seconds_after_backward-seconds_after_forward}")
                layers.reverse()
                self.optimiser.step(self.gradients)
                if correct:
                    total_correct = total_correct + 1
                # print(f"Time for image: {round(seconds_at_end_of_image - seconds_at_start_of_image, 5)}")
            average_loss = round(loss_total/len(training_images), 2)
            seconds_at_finish = time.time()

            average_correct = total_correct / len(training_images)
            print(f"Epoch {epoch}, loss = {average_loss}, time to run = {round((seconds_at_finish-seconds_at_start), 2)}, accuracy % = {round(average_correct * 100, 5)}")
            if average_loss < self.loss_best:
                self.loss_best = average_loss
                self.epoch_loss_best = epoch

            if epoch - self.epoch_loss_best > 5:
                break
        print(f"Best epoch = {self.epoch_loss_best}, with loss {self.loss_best}")

    def save_parameters(self):
        for layer in self.layers:
            filename = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\trained_model_data\layer_" + str(layer.id)
            with open(filename, "wb") as file:
                pickle.dump(layer, file)

    def load_parameters(self):
        layers_with_params = []
        for layer in layers:
            filename = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\trained_model_data\layer_" + str(layer.id)
            with open(filename, "rb") as file:
                layer = pickle.load(file)
                layers_with_params.append(layer)
        self.layers = layers_with_params
        return layers_with_params

    def accuracy_checK(self):
        print("Accuracy Check Started")
        correct_counter = 0
        for index in range (0, len(testing_images)):
            image = testing_images[index]
            label = testing_labels[index]
            forward = image
            for layer in layers:
                forward = layer.forward_pass(forward)
            prediction = numpy.argmax(forward)
            if prediction == label:
                correct_counter = correct_counter + 1
        print("Accuracy Check ended")
        return round((correct_counter / len(testing_images) * 100), 2)

    def get_prediction(self, image):
        forward = image
        for layer in layers:
            forward = layer.forward_pass(forward)
        prediction = numpy.argmax(forward)
        certainty = numpy.argmax(forward) #THIS CODE ISN'T FINISHED YET
        return prediction, certainty




classifier = Classification_Model(layers)
classifier.train()
# classifier.save_parameters()
# layers_with_params = classifier.load_parameters()
print(classifier.accuracy_checK())


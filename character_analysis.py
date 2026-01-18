
import tensorflow_datasets
import numpy
import math
import time
import pickle


datasets, dataset_info = tensorflow_datasets.load(
    "emnist/byclass",
    split=["train[:5]", "test[:1000]"],
    #Processing all of the data takes a fair bit of time. "m choosing to only load the first few samples for now and when I get to proper training obviously I'll load more.
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

    def forward_pass(self, image):
        padded = numpy.pad(image, ((self.padding, self.padding), (self.padding, self.padding), (0,0)), mode='constant')
        self.input = padded
        self.original_image = image
        filters = self.filters
        height, width, depth = padded.shape
        padding_px = height - self.input_width
        starting_pixel = 0 + padding_px//2
        activation_map = {}
        offset_size = (self.kernel_size - 1) // 2
        num_of_scores = (height - padding_px) // self.stride
        for filter in range (0, self.num_of_filters):
            scores = numpy.zeros((num_of_scores, num_of_scores))
            weights = filters[filter]["weights"]
            bias = filters[filter]["bias"]
            output_row = -1
            for row in range(starting_pixel, starting_pixel + self.input_width, self.stride):
                output_row = output_row + 1
                output_column = -1
                for column in range(starting_pixel, starting_pixel + self.input_width, self.stride):
                    output_column = output_column + 1
                    pixels_to_check = padded[row-offset_size: row+offset_size+1, column-offset_size: column+offset_size+1, :]
                    product = (weights * pixels_to_check)
                    score = numpy.sum(product) + bias
                    scores[output_row, output_column] = score

            activation_map[filter] = scores
        scores_3d = numpy.zeros((height-padding_px, width-padding_px, self.num_of_filters))
        for id in range(0, self.num_of_filters):
            scores_3d[:, :, id] = activation_map[id]
        return scores_3d

    def backprop(self, dOutput):
        dWeights = {}
        dBias = {}
        filters = self.filters
        dInput = numpy.zeros_like(self.input)
        for id, inner_dictionary in filters.items():
            total_dWeight = 0
            #Here, we want to get all 28*28 starting pixels, find dLoss/dWeights at this point, and then add it up.
            padded = self.input
            height, width, depth = padded.shape
            padding_px = height - self.input_width
            starting_pixel = 0 + padding_px // 2
            offset_size = (self.kernel_size - 1)//2
            output_row = -1
            for row in range (starting_pixel, starting_pixel+self.input_width, self.stride):
                output_row = output_row + 1
                output_column = -1
                for column in range(starting_pixel, starting_pixel+self.input_width, self.stride):
                    output_column = output_column + 1
                    row_start = row - offset_size
                    row_end = row + offset_size + 1
                    column_start = column - offset_size
                    column_end = column + offset_size + 1
                    pixels_to_check = padded[row_start:row_end, column_start:column_end, :]
                    grad = dOutput[output_row, output_column, id]
                    dWeight = grad * pixels_to_check
                    total_dWeight = total_dWeight + dWeight
            dBias = numpy.sum(dOutput[:, :, id])

            classifier.gradients[self.id][id]["weights"] = total_dWeight
            classifier.gradients[self.id][id]["bias"] = dBias




            gradient_filter = dOutput[:, :, id]
            height_gradients, width_gradients = gradient_filter.shape
            weights_for_filter = inner_dictionary["weights"]
            rotated_weights = numpy.rot90(weights_for_filter, 2)
            step_size = self.stride
            for grad_row in range (0, height_gradients):
                for grad_column in range (0, width_gradients):
                    input_row = starting_pixel + (grad_row * step_size)
                    input_column = starting_pixel +(grad_column * step_size)
                    grad_value = gradient_filter[grad_row, grad_column]
                    row_start = input_row - offset_size
                    row_end = input_row+ offset_size + 1
                    column_start = input_column - offset_size
                    column_end = input_column + offset_size + 1
                    dInput[row_start: row_end, column_start:column_end] = dInput[row_start:row_end, column_start:column_end] + grad_value * rotated_weights
        dInput = dInput[self.padding: -self.padding, self.padding: -self.padding, :]

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
                self.m[layer.id] = {}
                self.v[layer.id] = {}
                for filter in range(0, layer.num_of_filters):
                    self.m[layer.id][filter] = {
                        "weights": numpy.zeros_like(layer.filters[filter]["weights"]),
                        "bias": 0
                    }
                    self.v[layer.id][filter] = {
                        "weights": numpy.zeros_like(layer.filters[filter]["weights"]),
                        "bias": 0
                    }
        # for layer in layers:
        #     if layer.type == "CONV_Layer":
        #         self.gradients[layer.id] = {}
        #         for filter_id in range(layer.num_of_filters):
        #             self.gradients[layer.id][filter_id] = {
        #                 "weights": None,
        #                 "bias": None
        #             }
        #     else:
        #         self.gradients[layer.id] = {
        #             "weights": None,
        #             "bias": None
        #         }


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
                for filter in range(0, layer.num_of_filters):
                    m_weights = self.m[layer.id][filter]["weights"]
                    m_bias = self.m[layer.id][filter]["bias"]
                    v_weights = self.v[layer.id][filter]["weights"]
                    v_bias = self.v[layer.id][filter]["bias"]
                    dWeights = gradients[layer.id][filter]["weights"]
                    dBias = gradients[layer.id][filter]["bias"]

                    m_weights = self.beta1 * m_weights + (1 - self.beta1) * dWeights
                    v_weights = self.beta2 * v_weights + (1 - self.beta2) * (dWeights ** 2)
                    m_weights_protected = m_weights / (1 - self.beta1 ** self.timestep)
                    v_weights_protected = v_weights / (1 - self.beta2 ** self.timestep)
                    layer.filters[filter]["weights"] = layer.filters[filter]["weights"] - (self.learning_rate * m_weights_protected) / ( numpy.sqrt(v_weights_protected) + self.eps)
                    self.m[layer.id][filter]["weights"] = m_weights
                    self.v[layer.id][filter]["weights"] = v_weights

                    m_bias = self.beta1 * m_bias + (1 - self.beta1) * dBias
                    v_bias = self.beta2 * v_bias + (1 - self.beta2) * (dBias ** 2)
                    m_bias_protected = m_bias / (1 - self.beta1 ** self.timestep)
                    v_bias_protected = v_bias / (1 - self.beta2 ** self.timestep)
                    change = (self.learning_rate * m_bias_protected) / (numpy.sqrt(v_bias_protected) + self.eps)
                    layer.filters[filter]["bias"] = layer.filters[filter]["bias"] - change

                    self.m[layer.id][filter]["bias"] = m_bias
                    self.v[layer.id][filter]["bias"] = v_bias


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
                print(f"Forward for image {index} done")
                seconds_after_forward = time.time()
                print(f"time for forward = {seconds_after_forward - seconds_at_start_of_image}")
                loss, dLoss, correct = SVM_loss(forward, ground_truth)
                loss_total = loss_total + loss

                backward = dLoss
                layers.reverse()
                for layer in layers:
                    backward = layer.backprop(backward)
                print(f"Backprop for image {index} done")
                seconds_after_backward = time.time()
                print(f"time for backward = {seconds_after_backward-seconds_after_forward}")
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


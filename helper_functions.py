import cv2
import numpy
import math

def resize_to_28_x_28(numpy_array): #This function takes a numpy array of any size, rescales it to a size of 24*24 (maintaining aspect ratio), and pads it to make it 28*28
    height, width = numpy_array.shape
    biggest_size = max(height, width)
    scale_factor = 24/biggest_size
    new_component_height = int(height * scale_factor)
    new_component_width = int(width *scale_factor)
    resized = cv2.resize(numpy_array, (new_component_width, new_component_height), interpolation= cv2.INTER_LINEAR)
    x_padding = (28-new_component_width)//2
    y_padding = (28-new_component_height)//2
    x_remainder_padding = (28-new_component_width) % 2
    y_remaninder_padding = (28-new_component_height) % 2
    padded = numpy.pad(resized, ((y_padding, y_padding+y_remaninder_padding), (x_padding, x_padding+x_remainder_padding)), mode='constant', constant_values=0)
    return padded

def get_percentages_from_forward_pass(forward_pass_scores): #Given an output from the final linear layer, gets % certainties for each class
    max_score = numpy.max(forward_pass_scores)
    shifted = forward_pass_scores-max_score
    exp_scores = []
    for class_score in shifted: #This is the softmax algorithm.
        exp_score = math.e**class_score
        exp_scores.append(exp_score)
    sum_of_exps = sum(exp_scores)
    percentage_chances = []
    for exp in exp_scores:
        percentage_chance = exp/sum_of_exps
        percentage_chances.append(percentage_chance)
    return percentage_chances

def scale_array_to_0_to_1(numpy_array, inverse): #This scales a numpy array from 0-255 to 0-1
    if inverse:
        forward = 255 - numpy_array #This line is only necessary when creating the data from the maths dataset.
    else:
        forward = numpy_array
    scaled = numpy.divide(forward, 255)
    scaled_reshaped = numpy.reshape(scaled, (28, 28))
    return scaled_reshaped

def view_numpy_as_png(filepath, numpy_file, label): #This, given either a filepath where a npy is stored or a numpy array, shows it to the user.
    label_str = str(label)
    if filepath:
        image = numpy.load(filepath)
        cv2.imshow(label_str, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows() #Accesses the file at that filepath and shows it
    if numpy_file is not None:
        cv2.imshow(label_str, numpy_file)
        cv2.waitKey(0)
        cv2.destroyAllWindows() #Shows the file

def get_similar_letters(letter): #This takes an input of a letter and returns its 'confusables'
    letters_like_l = ["1", "ascii_124", "I", "!", "L"]
    letters_like_O = ["O", "0"]
    letters_like_c = [ "C", "("]
    letters_like_x = ["times", "X"]
    letters_like_F = ["F", "f"]
    groups = [letters_like_l, letters_like_O, letters_like_c, letters_like_x, letters_like_F]
    for group in groups:
        if letter in group:
            return group #If the letter is in one of the groups, returns the whole list of confusable letters
    return [letter] #Otherwise, returns just the letter (as a list)

def get_npy_images(components, npy_filename, numpy_array): #Given component information, this extracts the numpy arrays containing the components
    resized_list = []
    resized_ids_list = []
    for component in components:
        component_numpy_array = numpy_array[component.y: component.y + component.height, component.x: component.x + component.width]#Indexing the numpy file at the relevant positions
        resized = resize_to_28_x_28(component_numpy_array)
        resized_list.append(resized)
        resized_ids_list.append(component.id)
    return components, resized_list, npy_filename

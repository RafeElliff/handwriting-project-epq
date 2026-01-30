import cv2
import numpy


def resize_to_28_x_28(numpy_array):
    resized = cv2.resize(numpy_array, (24, 24), interpolation= cv2.INTER_LINEAR)
    padded = numpy.pad(resized, 2, mode='constant')
    return padded


def scale_array_to_0_to_1(numpy_array):
    scaled = numpy.divide(numpy_array, 255)
    scaled_reshaped = numpy.reshape(scaled, (28, 28))
    return scaled_reshaped

# def view_numpy_as_jpg(numpy_array):

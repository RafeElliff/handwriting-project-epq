import cv2
import numpy


def resize_to_28_x_28(numpy_array):
    resized = cv2.resize(numpy_array, (24, 24), interpolation= cv2.INTER_LINEAR)
    padded = numpy.pad(resized, ((2, 2), (2, 2)), mode='constant', constant_values=255)
    return padded


def scale_array_to_0_to_1(numpy_array, inverse):
    if inverse:
        forward = 255 - numpy_array #This line is only necessary when creating the data from the maths dataset.
    else:
        forward = numpy_array
    scaled = numpy.divide(forward, 255)
    scaled_reshaped = numpy.reshape(scaled, (28, 28))
    return scaled_reshaped

def view_numpy_as_jpg(filepath, numpy_file, label):
    if filepath:
        image = numpy.load(filepath)
        cv2.imshow(label, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if numpy_file is not None:
        cv2.imshow(label, numpy_file)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# dummy_array = numpy.random.randint(0, 256, size = (400, 400), dtype=numpy.uint8)
# padded = resize_to_28_x_28(dummy_array)
# scaled = scale_array_to_0_to_1(padded)
# view_numpy_as_jpg(filepath=None, numpy=scaled, label="Random")
# view_numpy_as_jpg(filepath=None, numpy=dummy_array, label="dummy")

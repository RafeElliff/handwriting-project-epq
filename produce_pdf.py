from reportlab.pdfgen import canvas
import os
from helper_functions import get_npy_images
import cv2
from segment_scans import full_segmentation_pipeline
from character_analysis import full_classification_pipeline
import numpy
base_pdf_folder= r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\final_pdfs"
images_prepared_base_folder = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\images\images_prepared_jpg"

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

#
original_page_width, original_page_height = (2000, 3000)
# c = canvas.Canvas(base_pdf_folder+"\\final_pdf.pdf", pagesize=(original_page_width,original_page_height))
# c.setFont("Times-Roman", 20)
# c.drawString(0, original_page_height-20, "Hello World")
file_name = "gold standard scan"
def draw_letters_to_pdf(letter_information_lists, file_name):
    images_prepared_filepath = os.path.join(images_prepared_base_folder, file_name+".jpg")
    original_page_height, original_page_width = cv2.imread(images_prepared_filepath, 2).shape
    print(original_page_height, original_page_width)
    c = canvas.Canvas(os.path.join(base_pdf_folder, file_name+".pdf"), pagesize=(original_page_width, original_page_height))
    for letter in letter_information_lists:
        letter_type = letter[0]
        letter_bottom_left = letter[1]
        letter_real_y = original_page_height - letter_bottom_left[1]
        letter_real_height = letter[2]
        letter_font_size = int(letter_real_height/0.7)
        c.setFont("Times-Roman", letter_font_size)
        c.drawString(letter_bottom_left[0], letter_real_y, str(letter_type))
    c.save()




letter_information_lists = [
    ('A', (70, 1000), 40),
    ('B', (140, 1000), 40),
    ('x', (210, 1040), 24),
    ('D', (280, 1000), 40),
    ('E', (350, 1000), 40),
    ('y', (420, 960), 20),
    ('F', (490, 1000), 40),
    ('G', (560, 1000), 40),
]

# letter_information_tuples = []
# for i in range(1, 61):
#     y_coordinate = 40*i
#     letter = ['a', (70, y_coordinate), 40]
#     letter_information_tuples.append(letter)

def get_letter_information_lists(filename):
    numpy_array, components, npy_filename = full_segmentation_pipeline(filename+".npy")
    components, list_of_resized, npy_filename = get_npy_images(components, npy_filename, numpy_array)
    numpy_list_of_resized = numpy.array(list_of_resized)
    normalised = numpy_list_of_resized/255
    predictions = full_classification_pipeline(normalised)
    letter_information_lists = []
    for index in range(0, len(components)):
        component = components[index]
        resized = list_of_resized[index]
        prediction = predictions[index]
        component_x_coordinate = component.x
        component_y_coordinate = component.y+component.height
        letter_information_list = [numbers_to_labels[prediction], (component_x_coordinate, component_y_coordinate), component.height]
        letter_information_lists.append(letter_information_list)
    return letter_information_lists

# draw_letters_to_pdf(letter_information_lists, file_name)
letter_information_lists = get_letter_information_lists(file_name)
draw_letters_to_pdf(letter_information_lists, file_name)
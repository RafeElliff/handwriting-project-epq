from reportlab.pdfgen import canvas
import os
from helper_functions import get_npy_images
import cv2
from segment_scans import full_segmentation_pipeline
from character_analysis import full_classification_pipeline
import numpy
from prepare_scans import get_skeletons
base_pdf_folder= r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\final_pdfs"
images_prepared_base_folder = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\images\images_lines_removed"

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

letter_type_to_letter = {
    '0': "0",
    '1': "1",
    '2': "2",
    '3': "3",
    '4': "4",
    '5': "5",
    '6': "6",
    '7': "7",
    '8': "8",
    '9': "9",
    'A': "A",
    'B': "B",
    'C': "C",
    'D': "D",
    'E': "E",
    'F': "F",
    'G': "G",
    'H': "H",
    'I': "I",
    'J': "J",
    'K': "K",
    'L': "L",
    'M': "M",
    'N': "N",
    'O': "O",
    'P': "P",
    'Q': "Q",
    'R': "R",
    'S': "S",
    'T': "T",
    'U': "U",
    'V': "V",
    'W': "W",
    'X': "X",
    'Y': "Y",
    'Z': "Z",
    'a': "a",
    'b': "b",
    'd': "d",
    'e': "e",
    'f': "f",
    'g': "g",
    'h': "h",
    'n': "n",
    'q': "q",
    'r': "r",
    't': "t",
    '!': "!",
    '(': "(",
    ')': ")",
    '+': "+",
    ',': ",",
    '-': "—",
    '=': "=",
    '[': "[",
    ']': "]",
    'alpha': "α",
    'ascii_124': "|",
    'beta': "β",
    'Delta': "Δ",
    'div': "÷",
    'exists': "∃",
    'forall': "∀",
    'forward_slash': "/",
    'gamma': "γ",
    'gt': ">",
    'in': "∈",
    'infty': "∞",
    'int': "∫",
    'lambda': "λ",
    'lt': "<",
    'mu': "μ",
    'neq': "≠",
    'phi': "ϕ",
    'pi': "π",
    'prime': "′",
    'rightarrow': "→",
    'sigma': "σ",
    'sqrt': "√",
    'sum': "Σ",
    'theta': "θ",
    'times': "×",
    '{': "{",
    '}': "}"
}


def get_standardised_info(letter_real_height, letter_real_width, letter_real_y):
    line_to_snap_to = round(letter_real_y/30, 0)
    y_to_snap_to = line_to_snap_to * 30
    # if letter_real_height < 20 and letter_real_width < 20:
    #     font_size = 50
    # else:
    #     font_size = 100
    average_proportion = max(letter_real_height, letter_real_width)
    font_size = int(average_proportion / 0.7)
    return y_to_snap_to, font_size
#
# c = canvas.Canvas(base_pdf_folder+"\\final_pdf.pdf", pagesize=(original_page_width,original_page_height))
# c.setFont("Times-Roman", 20)
# c.drawString(0, original_page_height-20, "Hello World")
file_name = "maths"
def draw_letters_to_pdf(letter_information_lists, file_name):
    images_prepared_filepath = os.path.join(images_prepared_base_folder, file_name+".png")
    original_page_height, original_page_width = cv2.imread(images_prepared_filepath, 2).shape
    print(original_page_height, original_page_width)
    c = canvas.Canvas(os.path.join(base_pdf_folder, file_name+".pdf"), pagesize=(original_page_width, original_page_height))
    for letter in letter_information_lists:
        letter_type = letter[0]
        letter_bottom_left = letter[1]
        letter_real_y = original_page_height - letter_bottom_left[1]
        letter_real_height = letter[2]
        letter_real_width = letter[3]
        y_to_snap_to, letter_font_size = get_standardised_info(letter_real_height, letter_real_width, letter_real_y)
        c.setFont("Times-Roman", letter_font_size)
        if letter_type == "-":
            letter_font_size = letter_font_size * 2
        c.drawString(letter_bottom_left[0], y_to_snap_to, letter_type_to_letter[str(letter_type)])
    c.save()


# c = canvas.Canvas(os.path.join(base_pdf_folder, "testing"+".pdf"), pagesize=(2000, 3000))
# starting_x = 0
# starting_y = 0
# for character_type, char_as_string in letter_type_to_letter.items():
#     starting_y = starting_y + 30
#     starting_x = starting_x + 20
#     c.setFont("Times-Roman", 50)
#     c.drawString(starting_x, starting_y, char_as_string)
# c.save()

# letter_information_tuples = []
# for i in range(1, 61):
#     y_coordinate = 40*i
#     letter = ['a', (70, y_coordinate), 40]
#     letter_information_tuples.append(letter)

def get_letter_information_lists(filename):
    components, normalised_skeletons = get_skeletons(filename)
    numpy_skeletons = numpy.array(normalised_skeletons)
    predictions = full_classification_pipeline(numpy_skeletons)
    letter_information_lists = []
    for index in range(0, len(components)):
        component = components[index]
        prediction = predictions[index]
        component_x_coordinate = component.x
        component_y_coordinate = component.y+component.height
        letter_information_list = [numbers_to_labels[prediction], (component_x_coordinate, component_y_coordinate), component.height, component.width]
        letter_information_lists.append(letter_information_list)
    return letter_information_lists

# draw_letters_to_pdf(letter_information_lists, file_name)
# letter_information_lists = get_letter_information_lists(file_name)
# draw_letters_to_pdf(letter_information_lists, file_name)
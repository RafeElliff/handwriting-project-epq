import math
import numpy
import cv2
from helper_functions import get_similar_letters, view_numpy_as_png, get_percentages_from_forward_pass
from load_images import get_EMNIST_images

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



def get_letter_possibilites(forward_pass_scores, percentage_chances, threshold_a=0.75, threshold_b=0.01, check_for_confusables=True):
    percentage_chances = get_percentages_from_forward_pass(forward_pass_scores)
    if numpy.max(percentage_chances) > threshold_a:
        return [numpy.argmax(percentage_chances)], percentage_chances
    else:
        potential_letters = []
        for index in range(len(percentage_chances)):
            class_percentage = percentage_chances[index]
            # print(class_percentage)
            if class_percentage > threshold_b:
                potential_letters.append(numbers_to_labels[index]) #Possible that using a system based on n highest scorers works better - need to test
        if check_for_confusables is True:
            first_letter = potential_letters[0]
            potential_letters_set = set(potential_letters)
            confusable_letters_for_letter = get_similar_letters(first_letter)
            confusable_letters_set = set(confusable_letters_for_letter)
            # print(potential_letters_set, confusable_letters_set)
            if potential_letters_set.issubset(confusable_letters_set):
                return [numpy.argmax(percentage_chances), percentage_chances]
            else:
                return potential_letters, percentage_chances
        else:
            return potential_letters, percentage_chances




def get_user_input(numpy_array, potential_letters, percentage_chances):
    image_scaled_up = (numpy_array * 255).astype(numpy.uint8)
    image_for_display = image_scaled_up[0, :, :, 0]
    if len(potential_letters) == 1:
        return potential_letters[0]
    else:
        sorted_letters = []
        for letter in potential_letters:
            letter_index = labels_to_numbers[letter]
            percentage = percentage_chances[letter_index]
            sorted_letters.append((percentage, letter))

        sorted_letters.sort(reverse=True)
        potential_letters = [letter for percentage, letter in sorted_letters]
        certainties = []
        for class_num in range(0, len(percentage_chances)):
            certainties.append((percentage_chances[class_num], class_num, numbers_to_labels[class_num]))
        certainties.sort(reverse=True)
        for index in range(0, min(len(certainties), 10)):
            percentage, class_num, label = certainties[index]
            if percentage > 0.01:
                print(f"{index + 1}. {label} : {round(percentage * 100, 2)}%")
        print("\n" * 3)

        cv2.imshow(str(potential_letters), image_for_display)
        while True:
            key_pressed_id = cv2.waitKey(0)
            key_pressed_char = int(chr(key_pressed_id))
            id_for_indexing = key_pressed_char - 1
            if id_for_indexing < len(potential_letters):
                cv2.destroyAllWindows()
                return labels_to_numbers[potential_letters[id_for_indexing]]
import os
import fitz
import cv2
import numpy
import segment_scans

onedrive_source = r'C:\Users\rafee\OneDrive\Scans'
images_pulled = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\images\images_pulled"
images_prepared_npy = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\images\images_prepared_npy"
images_prepared_jpg = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\images\images_prepared_jpg"
list_of_split_px_folder = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\images\list_of_split_px"


def copy_new_scans():
    for file_name in os.listdir(onedrive_source):  #loops through name of every file
        if file_name[-4:] == ".pdf":
            file_to_copy = os.path.join(onedrive_source, file_name)  #accesses the file with file_name
            dest_file = os.path.join(images_pulled, file_name[:-4] + ".jpg")
            if file_name[:-4] + ".jpg" not in os.listdir(images_pulled):  #checks that it hasn't already been copied
                pdf = fitz.open(file_to_copy)
                page = pdf[0]
                pix = page.get_pixmap(dpi=300)
                pix.save(dest_file)
                pdf.close()  #this converts the file to a jpg and saves it to images_pulled


def binarise_scan(source_jpg, file_name):
    grayscale_image = cv2.imread(source_jpg, 2)  #reads the image as a grayscale
    blurred = cv2.GaussianBlur(grayscale_image, (101, 101), 0)
    normalised = cv2.divide(grayscale_image, blurred, scale=255)
    _, black_and_white_image = cv2.threshold(normalised, 239, 255, cv2.THRESH_BINARY)  # stores the numpy datain black_and_white image
    black_and_white_image = cv2.bitwise_not(
        black_and_white_image)
    #the segmentation function used looks for white characters on a black background so it must be inverted
    numpy_array = black_and_white_image
    numpy_array = segment_scans.close_gaps(numpy_array)
    # numpy_array = segment_scans.morphological_opening(numpy_array)
    numpy.save(os.path.join(images_prepared_npy, file_name[:-4] + ".npy"), numpy_array)
    return numpy_array, file_name


def save_numpys():
    for file_name in os.listdir(images_pulled):
        if file_name not in os.listdir(images_prepared_npy):
            file_name_no_stem = file_name[:-4]
            source_jpg = os.path.join(images_pulled, file_name)
            binary_scan, file_name = binarise_scan(source_jpg, file_name)
            numpy_file, file_name = remove_lines(binary_scan, file_name)
            dest_file_npy = os.path.join(images_prepared_npy, file_name_no_stem)
            dest_file_jpg = os.path.join(images_prepared_jpg, file_name_no_stem)
            cv2.imwrite(dest_file_jpg + ".jpg", binary_scan)
            numpy.save(dest_file_npy + ".npy", numpy_file)


def remove_lines(numpy_file, file_name):
    height, width = numpy_file.shape
    line_starter_pixels = []
    all_line_px = set()
    letter_line_px = set()

    new_numpy_file_name = file_name[:-4] + "split_letter_px.npy"

    for y in range(1, height - 1):
        if numpy_file[y, 0] == 255:
            line_starter_pixels.append(y)
    existing_lines = []  #each 'line' is going to have the format of a list of [x coordinate, highest point, lowest point]
    for starter in line_starter_pixels:
        line_added = False
        for existing_line in existing_lines:
            if starter == existing_line[0] - 1:
                existing_line[0] = starter
                line_added = True
                break
            elif starter == existing_line[1] + 1:
                existing_line[1] = starter
                line_added = True
                break
        if line_added is False:
            existing_lines.append([starter, starter, 0])

    list_of_split_px = set()
    for line in existing_lines:  #I'm going to go back to storing pixels as tuples now. I didn't earlier because their immutability made it impossible with my algorithm, but I believe that tuples are neater
        highest_pixel = (line[0], 0)
        lowest_pixel = (line[1], 0)
        initial_line = [highest_pixel[0], lowest_pixel[0], 0]
        next_line = initial_line
        while next_line[2] < width - 1:
            next_line, part_of_letter, split_letter_px = find_next_highest_pixels(next_line,
                                                                                  numpy_file)  # I chose to break this up into multiple functions due to its complexity
            highest_pixel_y = next_line[0]
            lowest_pixel_y = next_line[1]
            if part_of_letter is True:
                for pixel in split_letter_px:
                    list_of_split_px.add(pixel)
            for pixel_y in range(highest_pixel_y, lowest_pixel_y + 1):
                all_line_px.add((int(pixel_y), int(next_line[2])))

    numpy_list_of_split_px = numpy.array(list(list_of_split_px))
    dest_file_npy = os.path.join(list_of_split_px_folder, new_numpy_file_name)
    numpy.save(dest_file_npy, numpy_list_of_split_px)
    for y_value in range(0, height):
        all_line_px.add((y_value, 0))
        all_line_px.add((y_value, width - 1))
    for x_value in range(0, width):
        all_line_px.add((0, x_value))
        all_line_px.add((height - 1, x_value))
    pixels_to_delete = all_line_px - letter_line_px
    for pixel in pixels_to_delete:
        numpy_file[pixel] = 0
        # pass
    return numpy_file, file_name


def find_next_highest_pixels(line, numpy_file):  #Given the highest and lowest pixel on a line in a column, this finds the highest and lowest pixel in the next column.
    height, width = numpy_file.shape
    line_highest = line[0]
    line_lowest = line[1]
    line_x = line[2]
    letter_above, letter_below, split_letter_px = search_for_letters(line, numpy_file)
    # Now, we check the three pixels adjacent to each of the top and bottom pixels, to find our new highest and lowest. The notation 'h', 's', 'l' means higher, same, lower i.e. highest_h would be the pixel that is one to the right and one above highest.
    highest_h = (max(line_highest - 1, 0), line_x + 1)
    highest_s = (line_highest, line_x + 1)
    highest_l = (min(line_highest + 1, height - 1), line_x + 1)  #each of these has the form (y, x)
    lowest_h = (max(line_lowest - 1, 0), line_x + 1)
    lowest_s = (line_lowest, line_x + 1)
    lowest_l = (min(line_lowest + 1, height - 1),
                line_x + 1)  # there's some unnecessary whitespace in this block, but it keeps the spacing consistent
    next_highest = None
    next_lowest = None
    if letter_above:
        next_highest = highest_s
    if letter_below:
        next_lowest = lowest_s
    if not letter_above:
        if numpy_file[highest_h] == 255:
            next_highest = highest_h
        elif numpy_file[highest_s] == 255:
            next_highest = highest_s
        elif numpy_file[highest_l] == 255:
            next_highest = highest_l
        else:
            for offset in range(2, 6):
                next_pixel = (min(line_lowest + offset, height-1), line_x + 1)
                if numpy_file[next_pixel] == 255:
                    next_highest = next_pixel
                    break
            if not next_highest:
                for offset in range(2, 4):
                    next_pixel = (max(0, line_lowest - offset), line_x + 1)
                    if numpy_file[next_pixel] == 255:
                        next_highest = next_pixel
                        break
            if not next_highest:
                next_highest = highest_s

    if not letter_below:
        if numpy_file[lowest_l] == 255:
            next_lowest = lowest_l
        elif numpy_file[lowest_s] == 255:
            next_lowest = lowest_s
        elif numpy_file[lowest_h] == 255:
            next_lowest = lowest_h
        else:
            for offset in range(2, 6):
                next_pixel = (max(0, line_lowest - offset), line_x + 1)
                if numpy_file[next_pixel] == 255:
                    next_lowest = next_pixel
                    break
            if not next_lowest:
                for offset in range(2, 4):
                    next_pixel = (min(line_lowest + offset, height-1), line_x + 1)
                    if numpy_file[next_pixel] == 255:
                        next_lowest = next_pixel
                        break
            if not next_lowest:
                next_lowest = lowest_s

    return_line = [next_highest[0], next_lowest[0], line_x + 1]
    part_of_letter = letter_below or letter_above
    return return_line, part_of_letter, split_letter_px


def search_for_letters(line, numpy_file):
    height, width = numpy_file.shape
    highest_pixel_y = line[0]
    highest_pixel_x = line[2]
    lowest_pixel_y = line[1]
    lowest_pixel_x = line[2]
    letter_above = False
    letter_below = False
    split_letter_px = []
    strip_to_check_upper = [
        (max(0, highest_pixel_y - 5), max(0, min(width - 6, highest_pixel_x - 6))),
        (max(0, highest_pixel_y - 5), max(0, min(width - 6, highest_pixel_x - 5))),
        (max(0, highest_pixel_y - 5), max(0, min(width - 6, highest_pixel_x - 4))),
        (max(0, highest_pixel_y - 5), max(0, min(width - 6, highest_pixel_x - 3))),
        (max(0, highest_pixel_y - 5), max(0, min(width - 6, highest_pixel_x - 2))),
        (max(0, highest_pixel_y - 5), max(0, min(width - 6, highest_pixel_x - 1))),
        (max(0, highest_pixel_y - 5), max(0, min(width - 6, highest_pixel_x))),
        (max(0, highest_pixel_y - 5), max(0, min(width - 6, highest_pixel_x + 1))),
        (max(0, highest_pixel_y - 5), max(0, min(width - 6, highest_pixel_x + 2))),
        (max(0, highest_pixel_y - 5), max(0, min(width - 6, highest_pixel_x + 3))),
        (max(0, highest_pixel_y - 5), max(0, min(width - 6, highest_pixel_x + 4))),
        (max(0, highest_pixel_y - 5), max(0, min(width - 6, highest_pixel_x + 5))),
        (max(0, highest_pixel_y - 5), max(0, min(width - 6, highest_pixel_x + 6))),
    ]
    arrow_to_check_upper = [
        (max(0, highest_pixel_y - 2), max(0, min(width - 2, highest_pixel_x))),
        (max(0, highest_pixel_y - 3), max(0, min(width - 2, highest_pixel_x + 1))),
        (max(0, highest_pixel_y - 3), max(0, min(width - 2, highest_pixel_x - 1))),
        (max(0, highest_pixel_y - 4), max(0, min(width - 2, highest_pixel_x + 2))),
        (max(0, highest_pixel_y - 4), max(0, min(width - 2, highest_pixel_x - 2)))
    ]
    pixels_to_check_upper = strip_to_check_upper + arrow_to_check_upper
    upper_foreground_px = 0
    for pixel in pixels_to_check_upper:
        if numpy_file[pixel] == 255:
            upper_foreground_px = upper_foreground_px + 1
            split_letter_px.append(pixel)
    if upper_foreground_px > 0:
        letter_above = True

    strip_to_check_lower = [
        (min(height - 1, lowest_pixel_y + 5), max(0, min(width - 6, lowest_pixel_x - 6))),
        (min(height - 1, lowest_pixel_y + 5), max(0, min(width - 6, lowest_pixel_x - 5))),
        (min(height - 1, lowest_pixel_y + 5), max(0, min(width - 6, lowest_pixel_x - 4))),
        (min(height - 1, lowest_pixel_y + 5), max(0, min(width - 6, lowest_pixel_x - 3))),
        (min(height - 1, lowest_pixel_y + 5), max(0, min(width - 6, lowest_pixel_x - 2))),
        (min(height - 1, lowest_pixel_y + 5), max(0, min(width - 6, lowest_pixel_x - 1))),
        (min(height - 1, lowest_pixel_y + 5), max(0, min(width - 6, lowest_pixel_x))),
        (min(height - 1, lowest_pixel_y + 5), max(0, min(width - 6, lowest_pixel_x + 1))),
        (min(height - 1, lowest_pixel_y + 5), max(0, min(width - 6, lowest_pixel_x + 2))),
        (min(height - 1, lowest_pixel_y + 5), max(0, min(width - 6, lowest_pixel_x + 3))),
        (min(height - 1, lowest_pixel_y + 5), max(0, min(width - 6, lowest_pixel_x + 4))),
        (min(height - 1, lowest_pixel_y + 5), max(0, min(width - 6, lowest_pixel_x + 5))),
        (min(height - 1, lowest_pixel_y + 5), max(0, min(width - 6, lowest_pixel_x + 6))),
    ]

    arrow_to_check_lower = [
        (min(height - 1, lowest_pixel_y + 2), max(0, min(width - 2, lowest_pixel_x))),
        (min(height - 1, lowest_pixel_y + 3), max(0, min(width - 2, lowest_pixel_x + 1))),
        (min(height - 1, lowest_pixel_y + 3), max(0, min(width - 2, lowest_pixel_x - 1))),
        (min(height - 1, lowest_pixel_y + 4), max(0, min(width - 2, lowest_pixel_x + 2))),
        (min(height - 1, lowest_pixel_y + 4), max(0, min(width - 2, lowest_pixel_x - 2)))
    ]

    pixels_to_check_lower = strip_to_check_lower + arrow_to_check_lower
    lower_foreground_px = 0
    for pixel in pixels_to_check_lower:
        if numpy_file[pixel] == 255:
            lower_foreground_px = lower_foreground_px + 1
            split_letter_px.append(pixel)

    if lower_foreground_px > 0:
        letter_below = True

    return letter_above, letter_below, split_letter_px

copy_new_scans()
save_numpys()
# remove_lines()

import os
import cv2
import numpy

images_binarised_npy = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\images\images_binarised_npy"
images_binarised_jpg = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\images\images_binarised_jpg"
list_of_split_px = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\images\list_of_split_px"


class Component:
    def __init__(self, x, y, width, height, area, centroid, id, split_letter):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.area = area
        self.centroid = centroid
        self.id = id
        self.split_letter = split_letter


def get_all_components(picture_name):
    image = numpy.load(os.path.join(images_binarised_npy, picture_name))  #loads the .npy file)
    num_of_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image,
                                                                               connectivity=8)  # connectivity refers to whether pixels diagonally adjacent are considered connected. On setting 8, they are.

    list_file_name = picture_name[:-4] + "split_letter_px.npy"
    split_letter_npy = numpy.load(os.path.join(list_of_split_px, list_file_name))

    min_size = 5 * 5
    max_size = 50 * 50  # if its larger or smaller its likely noise rather than a letter
    components_list = []
    split_letters = set()
    split_letters_px = set()
    for pixel in split_letter_npy:
        y, x = pixel
        if labels[y][x] != 0:
            split_letters.add(labels[y][x])

    for component in range(1, num_of_labels):
        x = stats[component, cv2.CC_STAT_LEFT]
        y = stats[component, cv2.CC_STAT_TOP]
        width = stats[component, cv2.CC_STAT_WIDTH]
        height = stats[component, cv2.CC_STAT_HEIGHT]
        area = stats[component, cv2.CC_STAT_AREA]
        centroid = centroids[component]
        id = component
        if id in split_letters:
            split_letter = True
        else:
            split_letter = False
        component = Component(x, y, width, height, area, centroid, id, split_letter)
        if is_valid_component(component):  #in order for it to be considered, it must meet these criteria. Otherwise, it is likely just noise or a leftover from the line removing algorithm.
            components_list.append(component)
    return components_list





def join_2_part_letters(components):  # this function joins the tittles in i and j to their stubs
    potential_stubs = []
    potential_tittles = []
    for component in components:
        if component.height / component.width > 1.6 and component.height < 65:  # if more than 1.6x as tall as wide, it's probably a stub
            potential_stubs.append(component)
        elif 20 < component.area < 120 :
            potential_tittles.append(component)  # if less than 60px in total, it's probably a tittle

    potential_stubs = sorted(potential_stubs, key=lambda component: component.centroid[0])
    potential_tittles = sorted(potential_tittles,
                               key=lambda component: component.centroid[0])  #Sorts them by their y centroid

    pairings = {}
    for stub in potential_stubs:
        for tittle in potential_tittles:
            if (
                    stub.x - 50 < tittle.x < stub.x + 50) and stub.y - 60 < tittle.y < stub.y:  # compares all tittles to the parameters they must meet to be joined
                pairings[stub] = tittle
                potential_tittles.remove(tittle)

    for stub, tittle in pairings.items():
        leftmost_point = min(stub.x, tittle.x)
        rightmost_point = max(stub.x + tittle.width, tittle.x + tittle.width)
        highest_point = min(stub.y, tittle.y)
        lowest_point = max(stub.y + stub.height, tittle.y + tittle.height)
        new_width = rightmost_point - leftmost_point
        new_height = lowest_point - highest_point  # working out the dimensions of the new component
        stub.x = leftmost_point
        stub.y = highest_point
        stub.width = new_width
        stub.height = new_height
        components.remove(
            tittle) # assigns the dimensions to the stub, removes the tittle, effectively combining them into one component.


    return components


def swap_x_y(coordinate):
    y = coordinate[0]
    x = coordinate[1]
    coordinate = (x, y)
    return coordinate


def match_split_components(potential_connections):

    all_components_in_connections = []
    for top, bottom in potential_connections:
        all_components_in_connections.append(top)
        all_components_in_connections.append(bottom)

    multi_partner_components = set()
    for component in all_components_in_connections:
        if all_components_in_connections.count(component) > 1:
            multi_partner_components.add(component)

    # Separate certain vs uncertain connections
    uncertain_connections = []
    final_connections = []

    for connection in potential_connections:
        top, bottom = connection
        if top in multi_partner_components or bottom in multi_partner_components:
            uncertain_connections.append(connection)
        else:
            final_connections.append(connection)

    matches = {}
    # for component in multi_partner_components:



def connect_split_letters(file_name):
    numpy_file = numpy.load(os.path.join(images_binarised_npy, file_name))
    numpy_file = morphological_opening(numpy_file)
    numpy_file = close_gaps(numpy_file)
    dest_file = os.path.join(images_binarised_npy, file_name)
    numpy.save(dest_file, numpy_file)

    components = get_all_components(file_name)
    components = join_2_part_letters(components)

    split_components = []
    for component in components:
        if component.split_letter is True and is_valid_component(component):
            split_components.append(component)

    split_components = sorted(split_components, key=lambda component: component.y)
    potential_connections = set()
    for component_top in split_components:
        top_L = component_top.x
        top_R = top_L + component_top.width
        top_T = component_top.y
        top_B = top_T + component_top.height
        for component_bottom in split_components:
            bottom_L = component_bottom.x
            bottom_R = bottom_L + component_bottom.width
            bottom_T = component_bottom.y
            bottom_B = bottom_T + component_bottom.height  # Working out the borders for the top and bottom components
            if top_B <= bottom_T and top_B + 10 >= bottom_T:
                if (top_L <= bottom_L <= top_R or bottom_L <= top_L <= bottom_R or top_L <= bottom_R <= top_R or bottom_L <= top_R <= bottom_R) and component_top != component_bottom:  # Making sure they are close enough to be joined
                    potential_connections.add((component_top, component_bottom))
    # potential_connections = match_split_components(raw_potential_connections)


    components = sorted(components, key=lambda component: component.id)
    num_of_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(numpy_file, connectivity=8)
    for connection in potential_connections:
        upper_component = connection[0]
        lower_component = connection[1]
        upper_component_bottom_row = upper_component.y + upper_component.height - 1  #if you didn't subtract 1, you'd be looking at the row beneath the bottom row
        lower_component_top_row = lower_component.y
        left_bottom = None  #This notation can be slightly confusing. The second half refers to the row, not the component, so left_bottom is the left pixel of the bottom row of the top component etc
        right_bottom = None
        left_top = None
        right_top = None
        for x_offset in range(0, upper_component.width):
            pixel_to_check = (upper_component_bottom_row, upper_component.x + x_offset)
            if numpy_file[pixel_to_check] == 255 and labels[pixel_to_check] == upper_component.id:
                if left_bottom is None:
                    left_bottom = pixel_to_check
                elif right_bottom is None:
                    right_bottom = pixel_to_check
                elif pixel_to_check[1] > right_bottom[1]:
                    right_bottom = pixel_to_check
        for x_offset in range(0, lower_component.width):
            pixel_to_check = (lower_component_top_row, lower_component.x + x_offset)
            if numpy_file[pixel_to_check] == 255 and labels[pixel_to_check] == lower_component.id:
                if left_top is None:
                    left_top = pixel_to_check
                elif right_top is None:
                    right_top = pixel_to_check
                elif pixel_to_check[1] > right_top[1]:
                    right_top = pixel_to_check
        #CV2 functions need coordinates as x,y
        if right_bottom and right_top and left_bottom and left_top:
            cv2.line(numpy_file, swap_x_y(right_bottom), swap_x_y(right_top), (255, 255, 255), 1)
            print(f"Drew a line from {swap_x_y(right_bottom)} to {swap_x_y(right_top)}")
            cv2.line(numpy_file, swap_x_y(left_bottom), swap_x_y(left_top), (255, 255, 255), 1)
            print(f"Drew a line from {swap_x_y(left_bottom)} to {swap_x_y(left_top)}")

            leftmost = min(left_bottom[1], left_top[1])
            rightmost = max(right_bottom[1], right_top[1])
            width_gap = rightmost - leftmost + 1
            height_gap = lower_component_top_row - upper_component_bottom_row
            for y in range(upper_component_bottom_row + 1, lower_component_top_row + 1):
                for x in range(leftmost + 1, rightmost + 1):
                    if numpy_file[y - 1][x] == 255 and numpy_file[y - 1][x + 1] == 255 and numpy_file[y][x - 1] == 255:
                        numpy_file[y][x] = 255

    components = get_all_components(file_name)
    components = join_2_part_letters(components)
    dest_file_npy = os.path.join(images_binarised_npy, file_name[:-4])
    dest_file_jpg = os.path.join(images_binarised_jpg, file_name[:-4])
    cv2.imwrite(dest_file_jpg + ".jpg", numpy_file)
    numpy.save(dest_file_npy + ".npy", numpy_file)
    return components

def is_valid_component(component):
    size = False
    area = False
    aspect_ratio = False
    if 5 < component.width < 75 and 5 < component.height < 90 or component.split_letter:
        size = True
    if 10 < component.area < 2500:
        area = True
    if 0.2 < component.width/component.height < 5:
        aspect_ratio = True
    return (size and area and aspect_ratio)

def close_gaps(numpy_file):
    kernel = numpy.ones((3, 3), numpy.uint8)
    closed_image = cv2.morphologyEx(numpy_file, cv2.MORPH_CLOSE, kernel)
    return closed_image

def morphological_opening(numpy_file):
    kernel = numpy.ones((3, 3), numpy.uint8)
    cleaned = cv2.morphologyEx(numpy_file, cv2.MORPH_OPEN, kernel)
    return cleaned

def find_component(components, id):
    for component in components:
        if component.id == id:
            return component


def look_through_npys():
    # loops through all npy files in the folder
    for filename in os.listdir(images_binarised_npy):
        file_path = os.path.join(images_binarised_npy, filename)
        # loads the file
        image = numpy.load(file_path)
        cv2.imshow(filename, image)
        cv2.waitKey(0)  #waits for a key press before moving
        cv2.destroyAllWindows()

def remove_extra_components(numpy_file):
    pass
# look_through_npys()


def show_components(image, components, scale=5):
    for i, c in enumerate(components):
        letter = image[c.y:c.y + c.height, c.x:c.x + c.width]

        # Enlarge for display
        letter_big = cv2.resize(
            letter,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST
        )

        cv2.imshow(f"Component {i}", letter_big)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# components_outer = connect_split_letters("standardise_1.npy")
# image = cv2.imread(os.path.join(images_binarised_jpg, "standardise_1.jpg"))
# components_outer = connect_split_letters("Scan - 2025-12-28 16_02_44.npy")
# image = cv2.imread(os.path.join(images_binarised_jpg, "Scan - 2025-12-28 16_02_44.jpg"))
# show_components(image, components_outer)

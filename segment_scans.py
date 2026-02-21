import os
import cv2
import numpy
from helper_functions import resize_to_28_x_28, view_numpy_as_png

images_heavily_binarised= r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\images\images_heavily_binarised"
images_weakly_binarised= r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\images\images_weakly_binarised"
list_of_split_px_folder = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\images\list_of_split_px"
images_lines_removed = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\images\images_lines_removed"
images_morphs_applied = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\images\images_morphs_applied"
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

def is_valid_component(component):
    size = False
    area = False
    aspect_ratio = False
    not_background = False
    if component.area < 50:
        return False
    print(component.x, component.y, component.id)
    if 5 < component.width <200 and 5 < component.height < 200:
        size = True
    else:
        print(f"Component {component.id} failed height/width check, dimensions = {component.height}, {component.width}")
    if 100 < component.area < 4000:
        area = True
    else:
        print(f"Component {component.id} failed area check, area = {component.area}")
    if 0.1 < component.width / component.height < 10:
        aspect_ratio = True
    else:
        print(f"Component {component.id} failed AR check, AR = {component.width/component.height}")
    if component.id != 0:
        not_background = True
    return (size and area and aspect_ratio and not_background)

def get_all_components(npy_array):
    num_of_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(npy_array,
                                                                               connectivity=8)  # connectivity refers to whether pixels diagonally adjacent are considered connected. On setting 8, they are.
    CCA_info = (num_of_labels, labels, stats, centroids)
    components_list = []

    for component in range(1, num_of_labels):
        x = stats[component, cv2.CC_STAT_LEFT]
        y = stats[component, cv2.CC_STAT_TOP]
        width = stats[component, cv2.CC_STAT_WIDTH]
        height = stats[component, cv2.CC_STAT_HEIGHT]
        area = stats[component, cv2.CC_STAT_AREA]
        centroid = centroids[component]
        id = component
        component = Component(x, y, width, height, area, centroid, id, False)
        components_list.append(component)
    return components_list, CCA_info



def join_2_part_letters(components, CCA_info, numpy_file):  # this function joins the tittles in i and j to their stubs
    labels = CCA_info[1]
    broken_components = []
    for component in components:
        if 50 < component.area < 500:
            broken_components.append(component)


    pairings = {}
    used_components = []
    for comp_a in broken_components:
        if comp_a not in used_components:
            for comp_b in broken_components:
                if comp_b not in used_components:
                    if (
                            comp_a.x - 20 < comp_b.x < comp_a.x + 20) and comp_a.y - 40 < comp_b.y < comp_a.y + 40 and comp_a != comp_b:  # compares all tittles to the parameters they must meet to be joined
                        pairings[comp_a] = comp_b
                        used_components.append(comp_a)
                        used_components.append(comp_b)

    for comp_a, comp_b in pairings.items():
        # print(len(pairings))
        leftmost_point = min(comp_a.x, comp_b.x)
        rightmost_point = max(comp_a.x + comp_b.width, comp_b.x + comp_b.width)
        highest_point = min(comp_a.y, comp_b.y)
        lowest_point = max(comp_a.y + comp_a.height, comp_b.y + comp_b.height)
        new_width = rightmost_point - leftmost_point
        new_height = lowest_point - highest_point  # working out the dimensions of the new component
        comp_a.x = leftmost_point
        comp_a.y = highest_point
        comp_a.width = new_width
        comp_a.height = new_height
        old_id = comp_b.id
        new_id = comp_a.id
        for y in range(comp_b.y, comp_b.y+comp_b.height):
            for x in range (comp_b.x, comp_b.x + comp_b.width):
                if labels[y][x] == old_id:
                    labels[y][x] = new_id
        components.remove(comp_b)
        # assigns the dimensions to the stub, removes the tittle, effectively combining them into one component.
    CCA_info = (CCA_info[0], labels, CCA_info[2], CCA_info[3])
    return components, CCA_info


def swap_x_y(coordinate):
    y = coordinate[0]
    x = coordinate[1]
    coordinate = (x, y)
    return coordinate

def connect_split_letters(numpy_file, second_image_to_draw_on, split_components):
    height, width = numpy_file.shape
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
                if (
                        top_L-2 <= bottom_L <= top_R+2 or bottom_L-2 <= top_L <= bottom_R+2 or top_L-2 <= bottom_R <= top_R+2 or bottom_L-2 <= top_R <= bottom_R+2) and component_top != component_bottom:  # Making sure they are close enough to be joined
                    potential_connections.add((component_top, component_bottom))
    # potential_connections = match_split_components(raw_potential_connections)

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
            cv2.line(numpy_file, swap_x_y(left_bottom), swap_x_y(left_top), (255, 255, 255), 1)
            cv2.line(second_image_to_draw_on, swap_x_y(right_bottom), swap_x_y(right_top), (255, 255, 255), 1)
            cv2.line(second_image_to_draw_on, swap_x_y(left_bottom), swap_x_y(left_top), (255, 255, 255), 1)
            leftmost = min(left_bottom[1], left_top[1])
            rightmost = max(right_bottom[1], right_top[1])
            width_gap = rightmost - leftmost + 1
            height_gap = lower_component_top_row - upper_component_bottom_row
            for y in range(upper_component_bottom_row + 1, lower_component_top_row + 1):
                for x in range(leftmost + 1, min(rightmost + 1, width-1)):
                    if numpy_file[y - 1][x] == 255 and numpy_file[y - 1][min(x + 1, width-1)] == 255 and numpy_file[y][x - 1] == 255:
                        numpy_file[y][x] = 255
                        second_image_to_draw_on[y][x] = 255

    return numpy_file, second_image_to_draw_on




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


# def look_through_npys():
#     # loops through all npy files in the folder
#     for filename in os.listdir(images_prepared_npy):
#         file_path = os.path.join(images_prepared_npy, filename)
#         # loads the file
#         image = numpy.load(file_path)
#         cv2.imshow(filename, image)
#         cv2.waitKey(0)  #waits for a key press before moving
#         cv2.destroyAllWindows()


def mark_split_line_components(components, CCA_info, filename):
    split_letter_npy = numpy.load(os.path.join(list_of_split_px_folder, filename[:-4] + "split_letter_px.npy"))
    labels = CCA_info[1]
    split_letter_ids = set()
    for pixel in split_letter_npy:
        y, x = pixel
        if labels[y][x] != 0:
            split_letter_ids.add(labels[y][x])
    return split_letter_ids

def clean_up_scan(components, CCA_info, numpy_file):
    labels = CCA_info[1]
    height, width = numpy_file.shape
    valid_components = set()
    # view_numpy_as_png(None, numpy_file, "test")
    for component in components:
        # print(component.x, component.y, is_valid_component(component), component.width, component.height, component.area)
        if is_valid_component(component):
            valid_components.add(component.id)

    new_components = []
    for component in components:
        if component.id in valid_components:
            new_components.append(component)
    for y in range (0, height):
        for x in range(0, width):
            if labels[y][x] not in valid_components:
                numpy_file[y][x] = 0
    return numpy_file, new_components

def remove_null_components(components, numpy_array):
    true_components = []
    for component in components:
        for y in range(component.y, component.y + component.height):
            for x in range(component.x, component.x + component.width):
                if numpy_array[y][x] == 255 and component not in true_components:
                    true_components.append(component)
                    break


    return true_components


def full_segmentation_pipeline(original_image, modified_image, filename):
    modified_image = close_gaps(modified_image)
    # view_numpy_as_png(None, modified_image, "test1")
    components, CCA_info = get_all_components(modified_image)
    split_letter_ids = mark_split_line_components(components, CCA_info, filename+".npy")
    split_components = set()
    for component in components:
        if component.id in split_letter_ids:
            component.split_letter = True
            split_components.add(component)

    modified_image, original_image = connect_split_letters(modified_image, original_image, split_components)
    # view_numpy_as_png(None, original_image, "test")
    # kernel = numpy.ones((5, 5), numpy.uint8)
    # dilated = cv2.dilate(modified_image, kernel, iterations=1)
    components, CCA_info = get_all_components(modified_image)
    components, CCA_info = join_2_part_letters(components, CCA_info, modified_image)
    # components, CCA_info = join_2_part_letters(components, CCA_info, modified_image)
    modified_image, components = clean_up_scan(components, CCA_info, modified_image)
    cv2.imwrite(os.path.join(images_morphs_applied, filename + ".png"), modified_image)
    components = remove_null_components(components, modified_image)
    return modified_image, original_image, components, filename








# cv2.imwrite(dest_file, numpy_array)
# show_components(numpy_array, components)

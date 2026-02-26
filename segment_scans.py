import os
import cv2
import numpy

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

def is_valid_component(component): #Checks if a component is valid
    size = False
    area = False
    aspect_ratio = False
    not_background = False
    if component.area < 50:
        return False
    #There are some print statements in this function that were used for testing: they are commented out but can be uncommented for debugging/interest
    # print(component.x, component.y, component.id)
    if 5 < component.width <200 and 5 < component.height < 200:
        size = True
    # else:
        # print(f"Component {component.id} failed height/width check, dimensions = {component.height}, {component.width}")
    if 100 < component.area < 4000:
        area = True
    # else:
        # print(f"Component {component.id} failed area check, area = {component.area}")
    if 0.1 < component.width / component.height < 10:
        aspect_ratio = True
    # else:
        # print(f"Component {component.id} failed AR check, AR = {component.width/component.height}")
    if component.id != 0:
        not_background = True
    return (size and area and aspect_ratio and not_background) #Returns true if it passes all of the criteria

def get_all_components(npy_array):
    #A component is a set of foreground pixels which are connected. The following function looks for them in a given numpy array.
    num_of_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(npy_array,
                                                                               connectivity=8)  # connectivity refers to whether pixels diagonally adjacent are considered connected. On setting 8, they are.
    CCA_info = (num_of_labels, labels, stats, centroids)
    components_list = []

    for component in range(1, num_of_labels): #Creates a component object for each component detected by the cv2 function
        x = stats[component, cv2.CC_STAT_LEFT]
        y = stats[component, cv2.CC_STAT_TOP]
        width = stats[component, cv2.CC_STAT_WIDTH]
        height = stats[component, cv2.CC_STAT_HEIGHT]
        area = stats[component, cv2.CC_STAT_AREA]
        centroid = centroids[component]
        id = component
        component = Component(x, y, width, height, area, centroid, id, False) #The split letter attribute is explained in the prepare_scans.py file
        components_list.append(component)
    return components_list, CCA_info

def join_2_part_letters(components, CCA_info, numpy_file): #The main function of this component in the current version of the code is to join the two bars in an = sign, but it will also function to join letters which should all be one component but have a slight gap due to thresholding/poor handwriting
    labels = CCA_info[1]
    broken_components = []
    for component in components:
        if 50 < component.area < 500:
            broken_components.append(component) #If the area is about these, then it makes sense that it might not be a whole letter


    pairings = {}
    used_components = []
    for comp_a in broken_components:
        if comp_a not in used_components:
            for comp_b in broken_components:
                if comp_b not in used_components:
                    if (
                            comp_a.x - 20 < comp_b.x < comp_a.x + 20) and comp_a.y - 40 < comp_b.y < comp_a.y + 40 and comp_a != comp_b:  #There's no rationale for these numbers, I just tested it and it seemed to work the best (increasing will join letters which shouldn't be joined, decreasing will
                        pairings[comp_a] = comp_b
                        used_components.append(comp_a)
                        used_components.append(comp_b)

    for comp_a, comp_b in pairings.items():
        #This block of code ombines both of the components into one singular component which has the dimensions to encompass both of the other components.
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
    CCA_info = (CCA_info[0], labels, CCA_info[2], CCA_info[3])
    return components, CCA_info

def swap_x_y(coordinate): #drawing in CV2 uses x,y as opposed to y,x like other parts of the program
    y = coordinate[0]
    x = coordinate[1]
    coordinate = (x, y)
    return coordinate

def connect_split_letters(numpy_file, second_image_to_draw_on, split_components):
    height, width = numpy_file.shape
    split_components = sorted(split_components, key=lambda component: component.y)
    potential_connections = set()
    # Working out the borders for the top and bottom components:
    for component_top in split_components:
        top_L = component_top.x
        top_R = top_L + component_top.width
        top_T = component_top.y
        top_B = top_T + component_top.height
        for component_bottom in split_components:
            bottom_L = component_bottom.x
            bottom_R = bottom_L + component_bottom.width
            bottom_T = component_bottom.y
            bottom_B = bottom_T + component_bottom.height
            if top_B <= bottom_T and top_B + 10 >= bottom_T:
                if (
                        top_L-2 <= bottom_L <= top_R+2 or bottom_L-2 <= top_L <= bottom_R+2 or top_L-2 <= bottom_R <= top_R+2 or bottom_L-2 <= top_R <= bottom_R+2) and component_top != component_bottom:  # Making sure they are close enough to be joined
                    potential_connections.add((component_top, component_bottom))


    num_of_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(numpy_file, connectivity=8)
    for connection in potential_connections:
        #This block of code finds which pixels to draw lines between
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
        if right_bottom and right_top and left_bottom and left_top:
            #Draws the relevant lines, and then fills in the gaps created in these lines
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

def close_gaps(numpy_file): #Applies a morphologcal close
    kernel = numpy.ones((3, 3), numpy.uint8)
    closed_image = cv2.morphologyEx(numpy_file, cv2.MORPH_CLOSE, kernel)
    return closed_image

def mark_split_line_components(components, CCA_info, filename):
    #For a given image, certain pixels will have been removed by the remove_lines function during preprocessing
    #This functions marks components to which this has happened
    split_letter_npy = numpy.load(os.path.join(list_of_split_px_folder, filename[:-4] + "split_letter_px.npy"))
    labels = CCA_info[1]
    split_letter_ids = set()
    for pixel in split_letter_npy:
        y, x = pixel
        if labels[y][x] != 0:
            split_letter_ids.add(labels[y][x])
    return split_letter_ids

def clean_up_scan(components, CCA_info, numpy_file): #Removes all invalid components (those which are too small
    labels = CCA_info[1]
    height, width = numpy_file.shape
    valid_components = set()
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

def remove_null_components(components, numpy_array): #During cleaning, some components become 'null' because all of their white pixels get removed. This removes the corresponding components from that list of components.
    true_components = []
    for component in components:
        for y in range(component.y, component.y + component.height):
            for x in range(component.x, component.x + component.width): #It does this by iterating through all of the pixels in the component: if a single pixel is foreground, then it is not 'null' and will be added to the safe components. Otherwise, it will be removed from the components list
                if numpy_array[y][x] == 255 and component not in true_components:
                    true_components.append(component)
                    break


    return true_components

def full_segmentation_pipeline(original_image, modified_image, filename):
    #From a preprocessed image:
    modified_image = close_gaps(modified_image) #Applies a close
    components, CCA_info = get_all_components(modified_image) #Gets all the components
    split_letter_ids = mark_split_line_components(components, CCA_info, filename+".npy") #Works out which compoentns were split, and joins them
    split_components = set()
    for component in components:
        if component.id in split_letter_ids:
            component.split_letter = True
            split_components.add(component)

    modified_image, original_image = connect_split_letters(modified_image, original_image, split_components)
    components, CCA_info = get_all_components(modified_image) #gets all components again (this is different to before because the components are joined now)
    components, CCA_info = join_2_part_letters(components, CCA_info, modified_image) #Combines the two part letters into a single component
    modified_image, components = clean_up_scan(components, CCA_info, modified_image)
    cv2.imwrite(os.path.join(images_morphs_applied, filename + ".png"), modified_image)
    components = remove_null_components(components, modified_image) #Cleans up the image, and returns the file
    return modified_image, original_image, components, filename
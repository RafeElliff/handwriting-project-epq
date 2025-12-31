import os
import cv2
import numpy

images_binarised_npy = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\images\images_binarised_npy"
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
    num_of_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)  # connectivity refers to whether pixels diagonally adjacent are considered connected. On setting 8, they are.

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
        if (3 < width < 70 and 5 < height < 70 and min_size < area < max_size) or split_letter is True: #in order for it to be considered, it must meet these criteria. Otherwise, it is likely just noise or a leftover from the line removing algorithm.
            components_list.append(Component(x, y, width, height, area, centroid, id, split_letter))
    return components_list


# i = 0
# for component in connected_components_analysis("Scan - 2025-12-28 16_02_44.npy"):
#     print(i, component.height/component.width)
#     i = i + 1
# print("components:", i)


def join_2_part_letters(components):  # this function joins the tittles in i and j to their stubs
    potential_stubs = []
    potential_tittles = []
    for component in components:
        if component.height / component.width > 1.6 and component.height < 40:  # if more than 1.6x as tall as wide, it's probably a stub
            potential_stubs.append(component)
        elif component.area < 60:
            potential_tittles.append(component)  # if less than 60px in total, it's probably a tittle
    potential_stubs = sorted(potential_stubs, key=lambda component: component.centroid[0])
    potential_tittles = sorted(potential_tittles, key=lambda component: component.centroid[0])  #Sorts them by their y centroid

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
        components.remove(tittle)  # assigns the dimensions to the stub, removes the tittle, effectively combining them into one component.

    return components


def connect_split_letters(file_name):
    list_file_name = file_name[:-4] + "split_letter_px.npy"
    # split_letter_npy = numpy.load(os.path.join(list_of_split_px, list_file_name))      # This is a list of all px which are thought to be part of a split letter.
    components = get_all_components(file_name)
    components = join_2_part_letters(components)
    split_components = []
    potential_connections = set()
    for component in components:
        if component.split_letter is True:
            split_components.append(component)
        if component.id in [17, 22, 60, 64, 35, 56, 62, 63]:
            print(component.x, component.y, component.id)



    split_components = sorted(split_components, key=lambda component: component.y)
    for component_top in split_components:
        top_L = component_top.x
        top_R = top_L + component_top.width
        top_T = component_top.y
        top_B = top_T + component_top.height
        for component_bottom in split_components:
            bottom_L = component_bottom.x
            bottom_R = bottom_L + component_bottom.width
            bottom_T = component_bottom.y
            bottom_B = bottom_T + component_bottom.height # Working out the borders for the top and bottom components
            if top_B <= bottom_T and top_B + 10 >= bottom_T:
                if bottom_L <= top_R + 10 and bottom_R + 10 >= top_L and component_top != component_bottom: # Making sure they are close enough to be joined
                    potential_connections.add((component_top.id, component_bottom.id))

        









def look_through_npys():
    # loops through all npy files in the folder
    for filename in os.listdir(images_binarised_npy):
        file_path = os.path.join(images_binarised_npy, filename)
        # loads the file
        image = numpy.load(file_path)
        cv2.imshow(filename, image)
        cv2.waitKey(0)  #waits for a key press before moving
        cv2.destroyAllWindows()


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


connect_split_letters("Scan - 2025-12-28 16_02_44.npy")
# image = numpy.load(os.path.join(images_binarised_npy, "Scan - 2025-12-28 16_02_44.npy"))
# show_components(image, join_2_part_letters(get_all_components("Scan - 2025-12-28 16_02_44.npy")))

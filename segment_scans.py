import os
import cv2
import numpy

images_binarised_npy = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\images\images_binarised_npy"


class Component:
    def __init__(self, x, y, width, height, area, centroid):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.area = area
        self.centroid = centroid


def connected_components_analysis(picture_name):
    image = numpy.load(os.path.join(images_binarised_npy, picture_name))  #loads the .npy file)
    num_of_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image,
                                                                               connectivity=8)  # connectivity refers to whether pixels diagonally adjacent are considered connected. On setting 8, they are.
    min_size = 5 * 5
    max_size = 50 * 50  # if its larger or smaller its likely noise rather than a letter
    components_list = []
    for component in range(1, num_of_labels):
        x = stats[component, cv2.CC_STAT_LEFT]
        y = stats[component, cv2.CC_STAT_TOP]
        width = stats[component, cv2.CC_STAT_WIDTH]
        height = stats[component, cv2.CC_STAT_HEIGHT]
        area = stats[component, cv2.CC_STAT_AREA]
        centroid = centroids[component]
        if 3 < width < 50 and 5 < height < 60 and min_size < area < max_size:
            components_list.append(Component(x, y, width, height, area, centroid))
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
        if component.height / component.width > 1.6 and component.height < 40:  # if more than 1.8x as tall as wide, it's probably a stub
            potential_stubs.append(component)
        elif component.area < 60:
            potential_tittles.append(component)  # if less than 60px in total, it's probably a tittle
    potential_stubs = sorted(potential_stubs, key=lambda component: component.centroid[0])
    potential_tittles = sorted(potential_tittles, key=lambda component: component.centroid[0])  #

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

def look_through_npys():
    # loops through all npy files in the folder
    for filename in os.listdir(images_binarised_npy):
        file_path = os.path.join(images_binarised_npy, filename)
        # loads the file
        image = numpy.load(file_path)
        cv2.imshow(filename, image)
        cv2.waitKey(0)  #waits for a key press before moving
        cv2.destroyAllWindows()


look_through_npys()


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

# image = numpy.load(os.path.join(images_binarised_npy, "Scan - 2025-12-28 16_02_44.npy"))
# show_components(image, join_2_part_letters(connected_components_analysis("Scan - 2025-12-28 16_02_44.npy")))

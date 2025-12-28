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
    image = numpy.load(os.path.join(images_binarised_npy, picture_name)) #loads the .npy file)
    num_of_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8) # connectivity refers to whether pixels diagonally adjacent are considered connected. On setting 8, they are.
    min_size = 5*5
    max_size = 50*50 # if its larger or smaller its likely noise rather than a letter
    components_list = []
    for component in range(1, num_of_labels):
        x = stats[component, cv2.CC_STAT_LEFT]
        y = stats[component, cv2.CC_STAT_TOP]
        width = stats[component, cv2.CC_STAT_WIDTH]
        height = stats[component, cv2.CC_STAT_HEIGHT]
        area = stats[component, cv2.CC_STAT_AREA]
        centroid = centroids[component]
        if 3 < width < 50 and 5 < height < 50 and min_size < area < max_size :
            components_list.append(Component(x, y, width, height, area, centroid))
    return components_list

i = 0
for component in connected_components_analysis("Scan - 2025-12-28 16_02_44.npy"):
    print(i, component.area, component.width, component.height, component.centroid)
    i = i + 1
print("components:", i)


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



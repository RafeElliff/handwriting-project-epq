import os
import fitz
import cv2
import numpy

onedrive_source = r'C:\Users\rafee\OneDrive\Scans'
images_pulled =r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\images\images_pulled"
images_binarised_npy = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\images\images_binarised_npy"
def copy_new_scans():
    for file_name in os.listdir(onedrive_source): #loops through name of every file
        if file_name[-4:] == ".pdf":
            file_to_copy = os.path.join(onedrive_source, file_name) #accesses the file with file_name
            dest_file = os.path.join(images_pulled, file_name[:-4]+".jpg")
            if file_name[:-4]+".jpg" not in os.listdir(images_pulled): #checks that it hasn't already been copied
                pdf = fitz.open(file_to_copy)
                page = pdf[0]
                pix = page.get_pixmap(dpi=300)
                pix.save(dest_file)
                pdf.close() #this converts the file to a jpg and saves it to images_pulled

def binarise_scans():
    for file_name in os.listdir(images_pulled):
        if file_name not in os.listdir(images_binarised_npy):
            colour_image = os.path.join(images_pulled, file_name)
            grayscale_image = cv2.imread(colour_image, 2) #reads the image as a grayscale
            _, black_and_white_image = cv2.threshold(grayscale_image, 80, 255, cv2.THRESH_BINARY) # stores the numpy datain black_and_white image
            dest_file = os.path.join(images_binarised_npy, file_name[:-4])
            numpy.save(dest_file, black_and_white_image) # stores numpy data in a .npy file in the images_binarised_npy folder


copy_new_scans()
binarise_scans()
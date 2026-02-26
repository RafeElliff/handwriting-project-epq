import os
import shutil

pdf_intermediate= r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\pdf_intermediate"
final_pdf_folder = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\final_pdfs"
final_locations = { #This can be customised to add different folders
    "m": r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\final_pdfs\maths",
    "c": r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\final_pdfs\chemistry",
    "p": r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\final_pdfs\physics",
    "M": r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\final_pdfs\maths",
    "C": r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\final_pdfs\chemistry",
    "P": r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\final_pdfs\physics",
}
uncategorised_destination = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\final_pdfs\other"

def get_pdf_destination_filepath(filename): #Works out which folder to assign the pdf to, and generates the corresponding filepath
    prefixes = final_locations.keys()
    if filename[1] in prefixes: #Activates if it has a special prefix with a folder assigned to it
        key = filename[1]
        destination_folder = final_locations[key]
        stripped_filename = filename[2:]
        destination_filepath = os.path.join(destination_folder, stripped_filename)
    else: # activates if the folder is unspecified - not one of the special keys
        stripped_filename = filename[2:]
        destination_filepath = os.path.join(uncategorised_destination, stripped_filename)
    return destination_filepath

def move_pdf(original_filename): #Copies a pdf from the pdf intermediate folded to its final destination folder
    original_filepath = os.path.join(pdf_intermediate, original_filename)
    destination_filepath = get_pdf_destination_filepath(original_filename)
    shutil.copy(original_filepath, destination_filepath)

def check_if_pdf_copied(pdf_filename): #Returns True if a pdf has already been made, to avoid remaking pdfs which have already been processed
    pdf_already_copied = False
    for folder in os.listdir(final_pdf_folder):
        for filename in os.listdir(os.path.join(final_pdf_folder, folder)):
            if filename == pdf_filename:
                pdf_already_copied = True
    return pdf_already_copied
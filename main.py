import os
pdf_intermediates = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\pdf_intermediate"
images_pulled = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\images\images_pulled"
from prepare_scans import copy_new_scans, save_numpys
from produce_pdf import get_letter_information_lists, draw_letters_to_pdf
from sort_intermediate_pdfs import move_pdf, check_if_pdf_copied

paper_type = "plain" #"plain" or "lined"
copy_new_scans() #Looks for any new pdfs in the scans folder, and transfers them to images_pulled
save_numpys(paper_type) #Undertakes the various preprocessing functions for all new files. For those where OCR does not occur, this function produces a pdf straight to pdf_intermediate

for png_filename in os.listdir(images_pulled):
    filename_no_ext = png_filename[:-4]
    pdf_filename = filename_no_ext + ".pdf"
    if pdf_filename not in os.listdir(pdf_intermediates):
        if pdf_filename[0] != "x" and pdf_filename != "X": #Does not run OCR if it is a file that is tagged as such
            letter_information_lists = get_letter_information_lists(filename_no_ext) #Gets information of what letters to draw
            draw_letters_to_pdf(letter_information_lists, filename_no_ext) # Draws letters

for pdf_intermediate in os.listdir(pdf_intermediates):
    if not check_if_pdf_copied(pdf_intermediate):
        move_pdf(pdf_intermediate) #Copies pdfs which have not yet been copied to their final destinations
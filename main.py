import os
pdf_intermediates = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\pdf_intermediate"
images_pulled = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\images\images_pulled"
from prepare_scans import copy_new_scans, save_numpys
from produce_pdf import get_letter_information_lists, draw_letters_to_pdf
from sort_intermediate_pdfs import move_pdf, check_if_pdf_copied

paper_type = "plain" #"plain" or "lined"
copy_new_scans()
save_numpys(paper_type)
for png_filename in os.listdir(images_pulled):
    filename_no_ext = png_filename[:-4]
    pdf_filename = filename_no_ext + ".pdf"
    if pdf_filename not in os.listdir(pdf_intermediates):
        print(pdf_filename)
        if pdf_filename[0] != "x" and pdf_filename != "X":
            letter_information_lists = get_letter_information_lists(filename_no_ext)
            draw_letters_to_pdf(letter_information_lists, filename_no_ext)

for pdf_intermediate in os.listdir(pdf_intermediates):
    if not check_if_pdf_copied(pdf_intermediate):
        move_pdf(pdf_intermediate)

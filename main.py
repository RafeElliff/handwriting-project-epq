import os
final_pdfs = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\final_pdfs"
images_pulled = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\images\images_pulled"
from prepare_scans import copy_new_scans, save_numpys
from produce_pdf import get_letter_information_lists, draw_letters_to_pdf


paper_type = "plain" #"plain" or "lined"
copy_new_scans()
save_numpys(paper_type)

for png_filename in os.listdir(images_pulled):
    filename_no_ext = png_filename[:-4]
    pdf_filename = filename_no_ext + ".pdf"
    if pdf_filename not in os.listdir(final_pdfs):
        letter_information_lists = get_letter_information_lists(filename_no_ext)
        draw_letters_to_pdf(letter_information_lists, filename_no_ext)
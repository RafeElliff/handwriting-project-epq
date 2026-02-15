import os
final_pdfs = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\final_pdfs"
images_pulled = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\images\images_pulled"
from prepare_scans import copy_new_scans, save_numpys
from segment_scans import full_segmentation_pipeline
from produce_pdf import get_letter_information_lists, draw_letters_to_pdf

copy_new_scans()
save_numpys()
for png_filename in os.listdir(images_pulled):
    pdf_filename = png_filename[:-4] + ".pdf"
    if pdf_filename not in os.listdir(final_pdfs):
        letter_information_lists = get_letter_information_lists(pdf_filename)
        draw_letters_to_pdf(letter_information_lists, pdf_filename)
import json

processed_images_json_file = final_pdf_folder = r"C:\Users\rafee\PycharmProjects\handwriting-project-epq\images\processed_images.json"

def check_if_image_processed(filename):
    with open (processed_images_json_file, "r") as file:
        processed_images = json.load(file)

    if filename in processed_images:
        return True
    else:
        return False

def mark_image_as_processed(filename):
    with open (processed_images_json_file, "r") as file:
        processed_images = json.load(file)
    processed_images.append(filename)
    with open (processed_images_json_file, "w") as file:
        json.dump(processed_images, file)


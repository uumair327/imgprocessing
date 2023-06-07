import cv2
import pytesseract
import os
from github import Github

# Function to perform Car Number Plate Recognition on a single image


def perform_anpr(image_path):
    # Load the car image
    image = cv2.imread(image_path)

    # Preprocess the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform number plate detection
    # ... (implement your number plate detection algorithm here)
    # You can use techniques like edge detection, contour detection, template matching, or deep learning-based methods for number plate detection

    # Extract the number plate region
    # ... (implement the extraction of number plate region here)
    # Once you have detected the number plate, you can extract the region using cropping or masking techniques

    # Apply OCR on the extracted number plate region
    number_plate_text = pytesseract.image_to_string(gray, config='--psm 7')

    return number_plate_text


# Function to save the processed image and extracted text to the GitHub repository
def save_output_to_github(repo_path, image_path, number_plate_text):
    # Load the image file from the local path
    with open(image_path, 'rb') as file:
        image_data = file.read()

    # Create a GitHub repository object
    g = Github("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")
    repo = g.get_repo(repo_path)

    # Upload the processed image to the repository
    image_filename = os.path.basename(image_path)
    try:
        # Check if the file already exists in the repository
        contents = repo.get_contents(image_filename)
        # Update the file using the sha parameter
        repo.update_file(contents.path, "ANPR output",
                         image_data, sha=contents.sha)
    except:
        # Create a new file if it doesn't exist
        repo.create_file(image_filename, "ANPR output", image_data)

    # Create a text file with the extracted number plate text
    text_filename = os.path.splitext(image_filename)[0] + ".txt"
    text_content = f"Number Plate Text: {number_plate_text}"
    try:
        # Check if the file already exists in the repository
        contents = repo.get_contents(text_filename)
        # Update the file using the sha parameter
        repo.update_file(contents.path, "ANPR output",
                         text_content, sha=contents.sha)
    except:
        # Create a new file if it doesn't exist
        repo.create_file(text_filename, "ANPR output", text_content)


# List of car images to process
car_images = [
    {
        "image_path": r"D:\Internship\Detection\try1.jpg",
        "output_folder": "output_folder_1"
    },
    {
        "image_path": r"D:\Internship\Detection\try2.jpg",
        "output_folder": "output_folder_2"
    },
    # Add more car images as needed
]

# Process each car image and save the outputs
for car_image in car_images:
    image_path = car_image["image_path"]
    output_folder = car_image["output_folder"]

    # Perform Car Number Plate Recognition
    number_plate_text = perform_anpr(image_path)

    # Save the processed image and extracted text to the output folder
    os.makedirs(output_folder, exist_ok=True)
    processed_image_path = os.path.join(output_folder, "processed_image.jpg")
    image = cv2.imread(image_path)
    cv2.imwrite(processed_image_path, image)
    text_file_path = os.path.join(output_folder, "output.txt")
    with open(text_file_path, 'w') as file:
        file.write(f"Number Plate Text: {number_plate_text}")

    # Save the processed image and extracted text to the GitHub repository
    repo_path = "uumair327/imgprocessing"  # Update the GitHub repository path
    save_output_to_github(repo_path, processed_image_path, number_plate_text)

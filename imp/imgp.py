import cv2
import numpy as np
from PIL import Image

# Step 1: Read image
image_path = "D:\Internship\imp\img1.png"  # Update the image path here

image = cv2.imread(image_path)

# Step 2: Show image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Rest of the code...

# Step 8: Face detection on the given image
# Update the cascade path here
cascade_path = "D:\Internship\imp\haarcascades-20230607T071315Z-001\haarcascades\haarcascade_eye.xml"

face_cascade = cv2.CascadeClassifier(cascade_path)
faces = face_cascade.detectMultiScale(
    image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Rest of the code...

# Step 9: Read & Save the video given
video_path = "D:\Internship\imp\vid1.mp4"  # Update the video path here

cap = cv2.VideoCapture(video_path)

# Rest of the code...

# Function to read and display the image


def read_and_show_image(image_path):
    image = cv2.imread(image_path)
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to convert image into Gray, Blur, Canny, Dilation & Eroded image


def image_transformations(image_path):
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Image", gray)
    cv2.waitKey(0)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    cv2.imshow("Blurred Image", blurred)
    cv2.waitKey(0)

    # Apply Canny edge detection
    edges = cv2.Canny(image, 100, 200)
    cv2.imshow("Canny Edges", edges)
    cv2.waitKey(0)

    # Apply dilation
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=1)
    cv2.imshow("Dilated Image", dilated)
    cv2.waitKey(0)

    # Apply erosion
    eroded = cv2.erode(image, kernel, iterations=1)
    cv2.imshow("Eroded Image", eroded)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

# Function to resize image


def resize_image(image_path):
    image = cv2.imread(image_path)

    # Resize smaller than the original
    smaller = cv2.resize(image, (300, 300))
    cv2.imshow("Smaller Image", smaller)
    cv2.waitKey(0)

    # Resize up to screen size
    screen_size = (1280, 720)  # Example screen size
    resized = cv2.resize(image, screen_size)
    cv2.imshow("Resized Image", resized)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

# Function to crop image


def crop_image(image_path):
    image = cv2.imread(image_path)

    # Define the region of interest
    x, y, w, h = 100, 100, 300, 300
    cropped = image[y:y+h, x:x+w]
    cv2.imshow("Cropped Image", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to insert shapes (rectangle & circle) in the given image


def insert_shapes(image_path):
    image = cv2.imread(image_path)

    # Draw a rectangle
    cv2.rectangle(image, (100, 100), (400, 300), (0, 255, 0), 2)

    # Draw a circle
    cv2.circle(image, (250, 200), 100, (0, 0, 255), 2)

    cv2.imshow("Image with Shapes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to join two & three different images


def join_images(image1_path, image2_path, image3_path):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    image3 = cv2.imread(image3_path)

    # Join two images horizontally
    joined1 = np.hstack((image1, image2))
    cv2.imshow("Joined Images 1", joined1)
    cv2.waitKey(0)

    # Join three images vertically
    joined2 = np.vstack((image1, image2, image3))
    cv2.imshow("Joined Images 2", joined2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function for face detection


def face_detection(image_path, cascade_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the Haar cascade XML file for face detection
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Perform face detection
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to read and save video


def read_and_save_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create VideoWriter object to save the video
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(
        *'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform any required image processing on the frame
        # ...

        # Write the processed frame to the output video
        out.write(frame)

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Example usage
image_path = "D:\Internship\imp\img1.png"
video_path = "D:\Internship\imp\vid1.mp4"
cascade_path = "D:\Internship\imp\haarcascades-20230607T071315Z-001\haarcascades\haarcascade_eye.xml"

read_and_show_image(image_path)
image_transformations(image_path)
resize_image(image_path)
crop_image(image_path)
insert_shapes(image_path)

image1_path = "path/to/your/image1.jpg"
image2_path = "path/to/your/image2.jpg"
image3_path = "path/to/your/image3.jpg"
join_images(image1_path, image2_path, image3_path)

face_detection(image_path, cascade_path)
read_and_save_video(video_path)

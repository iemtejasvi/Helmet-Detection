import cv2  # we're using OpenCV to mess with images.
import os  # This lets us navigate and use the operating system's functionalities.
import sys  # System stuff, mainly to grab command-line arguments.
import csv  # Helps us read and write CSV files easily.
from ultralytics import YOLO  # Grabbing YOLO from Ultralytics to detect stuff in images.

def imageLoader(folder_path):
    # Get a list of everything in the folder.
    items = os.listdir(folder_path)
    # Now filter this list to only keep images (look for .png, .jpg, .jpeg).
    images = [item for item in items if item.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # Quick print to tell us how many images we found.
    print(f"[!] Found {len(images)} images [!]")
    # Create full paths to the files so we can open them later.
    images_path_list = [os.path.join(folder_path, image) for image in images]
    return images_path_list, images

def saveResultCSV(result, output_folder_name, csv_file_name):
    # Make a full path for our new CSV file.
    csv_path = os.path.join(output_folder_name, csv_file_name + ".csv")
    # Open this new CSV file we're creating, get ready to write into it.
    with open(csv_path, "w", newline='') as f1:
        writer = csv.writer(f1, delimiter=",")
        # Writing the header row in the CSV.
        writer.writerow(["Image Name", "Image Location", "Status"])
        # Now, write each line of our results into the file.
        for row in result:
            writer.writerow(row)

def draw_bounding_box(image, bbox, label, confidence):
    # Convert bbox coordinates to integers, because pixels can't be fractions.
    x1, y1, x2, y2 = map(int, bbox)
    # Pick a color: green for 'helmet', red for everything else.
    color = (0, 255, 0) if 'helmet' in label.lower() else (0, 0, 255)
    # Draw the box on the image with the chosen color and a thickness of 2.
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    # Format the label with its confidence score to show on the image.
    label_with_conf = f"{label} ({confidence:.2f})"
    # Stick the label text on the image, slightly above the box.
    cv2.putText(image, label_with_conf, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def processDetections(results, image_name, image_path, image, csv_result_msg_final, image_storage_folder):
    # Default status, assuming we don't find a helmet.
    status = "No Helmet"
    # If we got no results back, just say so and bail.
    if not results:
        print("No results found.")
        return csv_result_msg_final
    
    # Let's just deal with the first result.
    result = results[0]
    # Convert the box coordinates into a format we can use (numpy array).
    boxes = result.boxes.numpy()
    # Get the labels for what was detected.
    labels = result.names

    # Loop through each detected object.
    for box, conf, cls_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
        label = labels[int(cls_id)]
        # We only care if the confidence is higher than 25%.
        if conf > 0.25:
            if 'helmet' in label.lower():
                status = "Helmet"  # Oh look, a helmet!
            # Draw the box around whatever we found.
            draw_bounding_box(image, box, label, conf)

    # Where we're gonna save the processed image.
    image_loc = os.path.join(image_storage_folder, image_name)
    # Save the image to that location.
    cv2.imwrite(image_loc, image)
    # Keep track of what we found in this image.
    csv_result_msg_final.append([image_name, image_path, status])
    return csv_result_msg_final

def processImages(image_path_list, image_name_list, image_storage_folder, model):
    # This is where we'll store our CSV results.
    csv_result_msg_final = []
    # Process each image one by one.
    for i, image_path in enumerate(image_path_list):
        # Read the image from its path.
        image = cv2.imread(image_path)
        # Detect stuff in the image using YOLO.
        results = model(image)
        # Process what we detected and update our results.
        csv_result_msg_final = processDetections(results, image_name_list[i], image_path, image, csv_result_msg_final, image_storage_folder)
        print("Debug: Image processed.")  # Little message to let us know it's done.
    return csv_result_msg_final

if __name__ == "__main__":
    try:
        # Read the folder path from the command line input.
        folder_path = sys.argv[1]
        # Define where we'll put our results.
        output_folder_name = "Result"
        # Make sure the output folder exists.
        os.makedirs(output_folder_name, exist_ok=True)
        # Where we'll store the images after processing.
        image_storage_folder = os.path.join(output_folder_name, "images")
        # Make sure that folder exists too.
        os.makedirs(image_storage_folder, exist_ok=True)

        # Load images and get paths and names.
        image_path_list, image_name_list = imageLoader(folder_path)
        # Path to the model we're using.
        model_path = r"C:\Users\steja\Helmet-Detection\models\data.pt"
        # Load up YOLO with the specified model.
        model = YOLO(model_path)

        # Let's process all the images.
        result = processImages(image_path_list, image_name_list, image_storage_folder, model)
        # Save our findings to a CSV.
        saveResultCSV(result, output_folder_name, csv_file_name="detection_results")
        # Let everyone know where the images and CSV ended up.
        print(f"Images saved to '{image_storage_folder}'")
        print(f"CSV file generated and saved to '{output_folder_name}'")
    except Exception as error:
        # If something went wrong, don't crash. Just print out the error.
        print("An error occurred:", str(error))

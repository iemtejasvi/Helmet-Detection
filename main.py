import cv2  # For dealing with images
import supervision as sv  # Helping us in supervision tasks
import os  # For doing stuff with files and folders
from datetime import datetime  # For handling dates and times
import sys  # Helps us talk with the system

from utils.helperFunctions import *  # Some extra help from our friends
from ultralytics import YOLO  # Magic tool for detecting things

# Our special tool for spotting helmets
model = YOLO(r"C:\Users\steja\Helmet-Detection\models\data.pt")

# How big our pictures should be
frame_wid = 640
frame_hyt = 480


def processImages(image_path_list, image_name_list, image_storage_folder):
    """
    Let's process some pictures! We'll find helmets using a cool model, draw on the pictures,
    and save them in a special place.

    Args:
        - image_path_list: Where our pictures are.
        - image_name_list: Names of those pictures.
        - image_storage_folder: The special place where we keep the modified pictures.

    Returns:
        - csv_result_msg_final: Messages about what we found in each picture.
    """

    # Tool for drawing boxes around stuff
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=1)

    csv_result_msg_final = []

    for i in range(len(image_path_list)):
        frame = cv2.imread(image_path_list[i])

        # print("Before Compression")
        # show_file_size(image_path_list[i])

        # Making the picture just the right size
        image = cv2.resize(frame, (frame_wid, frame_hyt))

        # Using our special model to find helmets
        results = model(image)[0]
        detections = sv.Detections.from_yolov8(results)
        labels = [f"{model.model.names[class_id]}" for _, _, class_id, _ in detections]

        # Drawing boxes around the helmets we found
        image = box_annotator.annotate(
            scene=image,
            detections=detections
            # labels=labels
        )

        # Checking if helmets are worn properly and saving results
        csv_result_msg_final = checkHeads(
            labels,
            image_name_list,
            image_path_list,
            image,
            csv_result_msg_final,
            i,
            image_storage_folder,
        )

        # Show the picture with drawings
        # cv2.imshow("Helmet Detection", image)
        # if cv2.waitKey(1) == 27:
        #     break

    return csv_result_msg_final


if __name__ == "__main__":
    """
    Our main job is to find helmets in pictures. We'll figure out where the pictures are,
    process them, and save our findings.
    """

    try:
        # Finding out where our pictures are and where we want to save our findings
        inter_path = sys.argv[1:]
        real_path = ""
        for path in inter_path:
            real_path = real_path + path + " "

        folder_path = real_path.strip()
        split_list = folder_path.split("\\")
        output_folder_name = os.path.join("Result", split_list[-1])
        os.makedirs(output_folder_name)

        image_storage_folder = os.path.join(output_folder_name, "images")
        os.makedirs(image_storage_folder)

    except Exception as error:
        print("[!] Oops! Something went wrong! [!]")

    try:
        # Let's load our pictures and find those helmets
        image_path_list, image_name_list = imageLoader(folder_path)
        result = processImages(image_path_list, image_name_list, image_storage_folder)

        # Saving our findings in a special file
        saveResultCSV(result, output_folder_name, csv_file_name=split_list[-1])

        print(
            f"Yay! We saved our pictures in '{image_storage_folder}' and our findings in '{output_folder_name}'"
        )

    except Exception as error:
        # Cleaning up if something went wrong
        print("[!] Oops! Something went wrong while processing! [!]")
        print(f"Error: {error}")
        os.rmdir(image_storage_folder)
        os.rmdir(output_folder_name)

import cv2  # Importing the tool to work with images
import os  # Importing a tool to work with files and folders
import csv  # Importing a tool to work with CSV files


def show_file_size(file):
    """
    This function checks how big a file is and tells us in a way we understand.

    Args:
        - file (str): The name of the file we want to check.

    Returns:
        None
    """

    # Check how big the file is and convert it into a size we can understand
    file_size = os.path.getsize(file)
    file_size_mb = round(file_size / 1024, 2)  # Convert to megabytes

    # Tell us how big the file is in a way we understand
    print("File size is " + str(file_size_mb) + "MB")


def imageLoader(folder_path):
    """
    This function looks in a folder and tells us how many pictures are there and where they are.

    Args:
        - folder_path (str): The place where the pictures are.

    Returns:
        Tuple: A list of where all the pictures are and a list of their names.
    """
    # Look at what's inside the folder
    items = os.listdir(folder_path)
    print(f"[!] Found {len(items)} images [!]")

    images_path_list = []  # Making a list to keep where the pictures are
    # Go through each thing in the folder
    for image in items:
        # Find where the picture is
        item_path = os.path.join(folder_path, image)
        images_path_list.append(item_path)

    # Give us a list of where all the pictures are and their names
    return (images_path_list, items)


def saveResultCSV(result, output_folder_name, csv_file_name):
    """
    This function saves what we found into a special file called a CSV file.

    Args:
        - result (list): A list of what we found, like the name of the picture, where it is, and what we found in it.
        - output_folder_name (str): The name of the folder where we will save the special file.
        - csv_file_name (str): The name we want to give to the special file.

    Returns:
        None
    """

    # Make a special file name with the folder and the name we want
    csv_path = os.path.join(output_folder_name, csv_file_name + ".csv")
    with open(csv_path, "w") as f1:
        writer = csv.writer(f1, delimiter=",")  # We need to make a special writer for our special file
        writer.writerow(["Image Name", "Image Location", "Status"])  # We need to write what each part means
        # Go through everything we found and write it into the special file
        for i in range(len(result)):
            row = result[i]
            writer.writerow(row)


def checkHeads(
    labels,
    image_name_list,
    image_path_list,
    image,
    csv_result_msg_final,
    i,
    image_storage_folder,
):
    """
    This function looks at what's in a picture and tells us if it sees a head.

    Args:
        - labels (list): A list of what it found in the picture.
        - image_name_list (list): A list of all the pictures' names.
        - image_path_list (list): A list of where all the pictures are.
        - image (numpy.ndarray): The picture itself.
        - csv_result_msg_final (list): A list of what we found in the pictures.
        - i (int): Which picture we're looking at.
        - image_storage_folder (str): Where we're going to keep the pictures we find.

    Returns:
        list: A list of what we found after looking at the pictures.
    """

    # Look at what it found in the picture
    if "head" in labels:
        print("head found")
        # Remember where the picture is and what its name is
        image_name = f"{image_name_list[i]}"
        image_loc = os.path.join(f"{image_storage_folder}/", image_name)
        # Save the picture where we can find it later
        cv2.imwrite(image_loc, image)

        img_loc = image_path_list[i]  # Remember where we found the picture
        message = "No Helmet"  # Tell us what it found in the picture

        # Put what it found into a special list to remember
        csv_result_msg_final.append([image_name, img_loc, message])

    return csv_result_msg_final

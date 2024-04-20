import cv2
import os
import sys
import csv
from ultralytics import YOLO

def show_file_size(file):
    file_size = os.path.getsize(file)
    file_size_mb = round(file_size / 1024, 2)
    print("File size is " + str(file_size_mb) + "MB")

def imageLoader(folder_path):
    items = os.listdir(folder_path)
    images = [item for item in items if item.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"[!] Found {len(images)} images [!]")
    images_path_list = [os.path.join(folder_path, image) for image in images]
    return (images_path_list, images)

def saveResultCSV(result, output_folder_name, csv_file_name):
    csv_path = os.path.join(output_folder_name, csv_file_name + ".csv")
    with open(csv_path, "w", newline='') as f1:
        writer = csv.writer(f1, delimiter=",")
        writer.writerow(["Image Name", "Image Location", "Status"])
        for row in result:
            writer.writerow(row)

def checkHeads(labels, image_name_list, image_path_list, image, csv_result_msg_final, i, image_storage_folder):
    if "head" in labels:
        print("head found")
        image_name = f"{image_name_list[i]}"
        image_loc = os.path.join(image_storage_folder, image_name)
        cv2.imwrite(image_loc, image)
        message = "No Helmet"
        csv_result_msg_final.append([image_name, image_path_list[i], message])
    return csv_result_msg_final

def processImages(image_path_list, image_name_list, image_storage_folder, model):
    csv_result_msg_final = []
    frame_wid = 640
    frame_hyt = 480

    for i, image_path in enumerate(image_path_list):
        frame = cv2.imread(image_path)
        image = cv2.resize(frame, (frame_wid, frame_hyt))
        results = model(image)
        print("Debug: Processing results...")

        for result in results:
            detections = []
            # Access the bounding box data from the 'boxes' attribute
            if hasattr(result, 'boxes') and result.boxes is not None:
                for bbox_tensor in result.boxes.data:
                    if len(bbox_tensor) == 6:
                        x1, y1, x2, y2, conf, cls_id = bbox_tensor
                        if conf > 0.25:
                            label = result.names[int(cls_id)]
                            detections.append({'bbox': (x1, y1, x2, y2), 'label': label, 'confidence': conf})
                    else:
                        print(f"Debug: Unexpected bbox_tensor dimensions or missing data - {bbox_tensor}")

            # Annotation functions would go here
            labels = [det['label'] for det in detections]
            csv_result_msg_final = checkHeads(labels, image_name_list, image_path_list, image, csv_result_msg_final, i, image_storage_folder)
            print("Debug: Image processed.")

    return csv_result_msg_final

if __name__ == "__main__":
    try:
        inter_path = sys.argv[1:]
        folder_path = " ".join(inter_path).strip()
        output_folder_name = os.path.join("Result", os.path.basename(folder_path))
        os.makedirs(output_folder_name, exist_ok=True)
        image_storage_folder = os.path.join(output_folder_name, "images")
        os.makedirs(image_storage_folder, exist_ok=True)

        image_path_list, image_name_list = imageLoader(folder_path)
        model_path = r"C:\Users\steja\Helmet-Detection\models\data.pt"
        model = YOLO(model_path)
        result = processImages(image_path_list, image_name_list, image_storage_folder, model)
        saveResultCSV(result, output_folder_name, csv_file_name=os.path.basename(folder_path))
        print(f"Images saved to '{image_storage_folder}' \nCSV file generated saved to '{output_folder_name}'")
    except Exception as error:
        print("[!] An error occurred: ", str(error))
        if os.path.exists(image_storage_folder):
            os.rmdir(image_storage_folder)
        if os.path.exists(output_folder_name):
            os.rmdir(output_folder_name)

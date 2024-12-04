import cv2  # we're using OpenCV to mess with images.
import os  # This lets us navigate and use the operating system's functionalities.
import sys  # System stuff, mainly to grab command-line arguments.
import csv  # Helps us read and write CSV files easily.
from ultralytics import YOLO  # Grabbing YOLO from Ultralytics to detect stuff in images.

def imageLoader(folder_path):
    items = os.listdir(folder_path)
    images = [item for item in items if item.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"[!] Found {len(images)} images [!]")
    images_path_list = [os.path.join(folder_path, image) for image in images]
    return images_path_list, images

def saveResultCSV(result, output_folder_name, csv_file_name):
    csv_path = os.path.join(output_folder_name, csv_file_name + ".csv")
    with open(csv_path, "w", newline='') as f1:
        writer = csv.writer(f1, delimiter=",")
        writer.writerow(["Image Name", "Image Location", "Status"])
        for row in result:
            writer.writerow(row)

def draw_bounding_box(image, bbox, label, confidence):
    x1, y1, x2, y2 = map(int, bbox)
    color = (0, 255, 0) if 'helmet' in label.lower() else (0, 0, 255)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    label_with_conf = f"{label} ({confidence:.2f})"
    cv2.putText(image, label_with_conf, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def processDetections(results, image_name, image_path, image, csv_result_msg_final, image_storage_folder):
    status = "No Helmet"
    if not results:
        print("No results found.")
        return csv_result_msg_final

    result = results[0]
    boxes = result.boxes.cpu().numpy()  # Move to CPU and convert to NumPy
    labels = result.names

    for box, conf, cls_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
        label = labels[int(cls_id)]
        if conf > 0.25:
            if 'helmet' in label.lower():
                status = "Helmet"
            draw_bounding_box(image, box, label, conf)

    image_loc = os.path.join(image_storage_folder, image_name)
    cv2.imwrite(image_loc, image)
    csv_result_msg_final.append([image_name, image_path, status])
    return csv_result_msg_final

def processImages(image_path_list, image_name_list, image_storage_folder, model):
    csv_result_msg_final = []
    for i, image_path in enumerate(image_path_list):
        image = cv2.imread(image_path)
        results = model(image)
        csv_result_msg_final = processDetections(results, image_name_list[i], image_path, image, csv_result_msg_final, image_storage_folder)
        print("Debug: Image processed.")
    return csv_result_msg_final

if __name__ == "__main__":
    try:
        folder_path = sys.argv[1]
        output_folder_name = "Result"
        os.makedirs(output_folder_name, exist_ok=True)
        image_storage_folder = os.path.join(output_folder_name, "images")
        os.makedirs(image_storage_folder, exist_ok=True)

        image_path_list, image_name_list = imageLoader(folder_path)
        model_path = os.path.join(os.path.dirname(__file__), "models", "data.pt")
        model = YOLO(model_path)

        result = processImages(image_path_list, image_name_list, image_storage_folder, model)
        saveResultCSV(result, output_folder_name, csv_file_name="detection_results")
        print(f"Images saved to '{image_storage_folder}'")
        print(f"CSV file generated and saved to '{output_folder_name}'")
    except Exception as error:
        print("An error occurred:", str(error))


**Usage Guide for Helmet Detection Using YOLOv8**

Welcome to the Helmet Detection Using YOLOv8 project! This project utilizes the YOLOv8 model specifically trained for detecting helmets in images and videos. Below is a comprehensive guide on how to set up and use this project effectively.

### Setup

1. **Clone the Repository:**

   Clone the repository to your local machine using the following command:

   ```bash
   git clone https://github.com/iemtejas/Helmet-Detection.git
   ```

2. **Navigate to the Project Directory:**

   Once cloned, move into the project directory:

   ```bash
   cd Helmet-Detection
   ```

3. **Create and Activate Conda Environment:**

   Create a Conda environment and activate it:

   ```bash
   conda create -n helmet_detection python=3.9.1
   conda activate helmet_detection
   ```

4. **Install Requirements:**

   Install the required Python packages listed in the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

5. **Run the Helmet Detection Script:**

   Execute the main Python script (`main.py`) by providing the path to the directory containing your images or videos:

   ```bash
   python main.py /path/to/images_or_videos
   ```

   Replace `/path/to/images_or_videos` with the actual path to the directory containing the images or videos you want to perform helmet detection on.

### Model

The project utilizes YOLOv8, an advanced variant of the YOLO (You Only Look Once) object detection algorithm. This specific model is trained to detect helmets accurately in various scenarios.

### Limitations

- The effectiveness of helmet detection may vary depending on factors such as lighting conditions, helmet size, orientation, and occlusion.
- While YOLOv8 is known for its efficiency and accuracy, it may not perform optimally in all scenarios and may require fine-tuning on specific datasets for improved accuracy.
- Processing large videos or a large number of high-resolution images may require substantial computational resources.

### Contributing

Contributions to this project are welcome! If you have suggestions for improvements or want to contribute code, feel free to fork the repository, make your changes, and submit a pull request.

### Issues

If you encounter any issues while using the project or have suggestions for improvements, please open an issue on the GitHub repository. Your feedback is valuable in improving the project.

### License

This project is licensed under the [MIT License](LICENSE).


# Speed Estimation for Cars on a Highway

This Python script uses YOLO (You Only Look Once) model to estimate the speed of cars on a highway from a video file. The script processes the video, detects the cars, and estimates their speed based on the provided reference points.

## Prerequisites

- Python 3.x
- `ultralytics` library
- `opencv-python` library

You can install the required libraries using pip:

```sh
pip install ultralytics opencv-python
```

## Configuration

Before running the script, create a JSON configuration file (`config.json`) with the following structure:

```json
{
    "model_path": "yolov8s.pt",
    "video_path": "path/to/your/video.mp4",
    "output_path": "path/to/output/video.avi",
    "line_pts": [[0, 400], [1280, 400]]
}
```

- `model_path`: Path to the YOLO model file.
- `video_path`: Path to the input video file.
- `output_path`: Path where the output video will be saved.
- `line_pts`: Points defining the reference line for speed estimation.

## Running the Script

1. Save the Python script as `measuring_the_speed_of_cars.py`.

2. Create the configuration file (`config.json`) as shown above.

3. Run the script with the following command:

```sh
python measuring_the_speed_of_cars.py --config config.json
```

## Script Explanation

1. **Imports and Logger Setup**:
   - The script imports necessary libraries and sets up logging to track the script's execution.

2. **Main Function**:
   - The main function loads the configuration file and initializes the YOLO model.
   - It opens the input video and prepares the video writer for the output.
   - The speed estimation object is initialized with car names and reference points.

3. **Processing Loop**:
   - The script processes each frame of the video, tracks the cars, and estimates their speed.
   - The processed frame with speed annotations is written to the output video.

4. **Clean Up**:
   - The script releases the video capture and writer objects and closes any OpenCV windows.

## Error Handling

The script includes basic error handling to log any issues encountered during video processing.

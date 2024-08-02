from ultralytics import YOLO
from ultralytics.solutions import speed_estimation
import cv2

model = YOLO("yolov8s.pt")
names = model.model.names

cap = cv2.VideoCapture("cars_on_a_highway/los-angeles-freeway-i-101-hd-30fps-traffic-sound.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("cars_on_a_highway/speed_estimation.avi",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (w, h))

line_pts = [(0, 400), (1280, 400)]

speed_obj = speed_estimation.SpeedEstimator(names=names, reg_pts=line_pts, view_img=True)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    tracks = model.track(im0, persist=True, show=False)

    im0 = speed_obj.estimate_speed(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()

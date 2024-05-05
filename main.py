# import os
# import random

# import cv2
# from ultralytics import YOLO

# from tracker import Tracker
# import time


# # video_path = os.path.join('.', 'data', 'people.mp4')
# video_out_path = os.path.join('.', 'out.mp4')

# # cap = cv2.VideoCapture(video_path)
# cap = cv2.VideoCapture(0)
# ret, frame = cap.read()

# cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
#                           (frame.shape[1], frame.shape[0]))

# model = YOLO("yolov8n.pt")

# tracker = Tracker()

# colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

# detection_threshold = 0.5
# while ret:

#     results = model(frame)
#     yolo_frame_time = time.time() - tracker.yolo_start_time
#     yolo_fps = int(1 / yolo_frame_time)
#     cv2.putText(frame, f"YOLOv8 FPS: {yolo_fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
#     tracker.yolo_start

#     for result in results:
#         detections = []
#         for r in result.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = r
#             x1 = int(x1)
#             x2 = int(x2)
#             y1 = int(y1)
#             y2 = int(y2)
#             class_id = int(class_id)
#             if score > detection_threshold:
#                 detections.append([x1, y1, x2, y2, score])

#         tracker.update(frame, detections)

#         for track in tracker.tracks:
#             bbox = track.bbox
#             x1, y1, x2, y2 = bbox
#             track_id = track.track_id

#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
#             # cv2.putText(frame, "ID: " + str(track_id), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (colors[track_id % len(colors)]), 2)


#     cv2.imshow('Output', frame)
#     cap_out.write(frame)
#     ret, frame = cap.read()
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break 
  

# cap.release()
# cap_out.release()
# cv2.destroyAllWindows()

import os
import random
import cv2
import time
from ultralytics import YOLO
from tracker import Tracker


video_path = os.path.join('.', 'data', 'test.mp4')
video_out_path = os.path.join('.', 'out.mp4')

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

model = YOLO("yolov8n.pt")

tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]

detection_threshold = 0.5
while ret:
    deepsort_frame_start_time = time.time()

    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
            cv2.putText(frame, str(track_id), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (colors[track_id % len(colors)]), 2)

    # Calculate DeepSORT FPS based on the time taken for each frame
    deepsort_frame_time = time.time() - deepsort_frame_start_time
    deepsort_fps = int(1 / deepsort_frame_time)

    # Display DeepSORT FPS in the frame
    cv2.putText(frame, f"DeepSORT FPS: {deepsort_fps}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Output', frame)
    cap_out.write(frame)
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cap_out.release()
cv2.destroyAllWindows()


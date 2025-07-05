import cv2
import math
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    results = model(frame, stream = True)

    for r in results:
        class_names  = r.names
        print(class_names)
        boxes = r.boxes

        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]

            class_id = int(box.cls[0])  # extracts the id, convert it into int() else it will be in tensor format i.e.: tensor(0.)
            obj_name = class_names[class_id]  # using that id, we find the name

            confidence_score = box.conf[0]

    #        print(confidence_score)

            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1,y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f'{obj_name}-{math.ceil((confidence_score * 100))/100}', (max(20, x1), max(20, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)  # (max(20, x1), max(20, y1)) now even if the object goes out of the frame, the confidence score will still stays on the frame.
    
    cv2.imshow("Frames", frame)
    cv2.waitKey(1)
import cv2
import math
import numpy as np
from sort import *
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(r'cars.mp4')

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mask_img = cv2.imread(r'C:\Users\hp\OneDrive\Desktop\computer vision\ComputerVisionVersion2\mask.png')
print(mask_img.shape)
# TRACKING
tracker = Sort(max_age = 20, min_hits = 3, iou_threshold = 0.3)

counter = 0
total_counts = []
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (720, 480))

    detections = np.empty((0,5))

    print(frame.shape)
    if not ret: 
        break
    
    # img_region = cv2.bitwise_and(frame, mask_img)

    results = model(frame, stream = True)

    for r in results:
        class_names  = r.names
        # print(class_names)
        boxes = r.boxes

        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
    
            class_id = int(box.cls[0])  # extracts the id, convert it into int() else it will be in tensor format i.e.: tensor(0.)
            obj_name = class_names[class_id]  # using that id, we find the name

            confidence_score = box.conf[0]
            # print(confidence_score)

            if obj_name == "truck" and confidence_score > 0.25:
                cv2.rectangle(frame, (x1,y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, f'{obj_name}', (max(20, x1), max(20, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 3)  # (max(20, x1), max(20, y1)) now even if the object goes out of the frame, the confidence score will still stays on the frame.
                # cv2.putText(frame, f'{obj_name}-{math.ceil((confidence_score * 100))/100}', (max(20, x1), max(20, y1)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 2)  # (max(20, x1), max(20, y1)) now even if the object goes out of the frame, the confidence score will still stays on the frame.
                currentArray = np.array([x1, y1, x2, y2, confidence_score])
                detections = np.vstack((detections, currentArray))
                
        results_tracker = tracker.update(detections)
        
        # THIS LINE IS TO DECIDE THE LINE TO COUNT
        cv2.line(frame, (130, 300), (660, 300), (0,255,0),4)  

        for result in results_tracker:

            print(x1, y1, x2, y2, id)
            x1, y1, x2, y2, id = result   # these are the floting numbers
            x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
            h, w = y2-y1, x2-x1

            print(result)

            cx = x1 + w //2 
            cy = y1 + h //2 

            cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if 120 < cx < 670 and 285 < cy < 315:
                if total_counts.count(id) == 0:
                    total_counts.append(id)
    cv2.putText(frame, f'COUNTER: {len(total_counts)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 1), 2) 

    cv2.imshow("Frames", frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
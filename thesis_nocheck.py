from ultralytics import YOLO
# from picamera import PiCamera
from time import sleep
import time
import numpy as np
import cv2
# from gpiozero import Button

# button = Button(5)

model = YOLO("C:/Users/zhakl/Desktop/Thesis/Thesis/weights/lane.pt")

# camera = PiCamera()
points_list = []
capture = cv2.VideoCapture("utube.mp4")
success, frame = capture.read()

minX = []
minY = []
maxX = []
maxY = []
count = 0
plate = YOLO("C:/Users/zhakl/Desktop/Thesis/Thesis/weights/plate.pt")

vehicle = YOLO("yolov8n.pt")

def detect_collision(x, y):
    global minX
    global minY
    global maxX
    global maxY
    global points_list
    dist = -1
    for i, points in enumerate(points_list):
        # print("asd")
        if x > maxX[i] or x < minX[i]:
            continue
        if y > maxY[i] or y < minY[i]:
            continue        
        dist = cv2.pointPolygonTest(points, (int(x),int(y)), False)
        if dist > 0:
            return dist
    return dist

def plate_detection(img):
    global plate
    global count
    results = plate(img, save=True) 
    boxes = results[0].boxes.xyxy.tolist()
    for box in boxes:
        object = [int(box[0]), int(box[1]), int(box[2]- box[0]), int(box[3] - box[1])]
        x,y,w,h = object   
        cv2.imwrite('detected_plate_' + str(count) + '.jpg', img[y:y+h, x:x+w])
        count += 1

def process_video():
    global model
    # global capture
    global success
    n = 0
    # camera.start_recording("/home/zhaklee/Desktop/thesis/redlight.h264")
    capture = cv2.VideoCapture("utube.mp4")
    success,image = capture.read()
    start = time.time()
    while success:
        success,image = capture.read()
        n += 1
        if n % 20 == 0:
            result = model(image, save=True, classes=[7])
            res_boxes = result[0].boxes.xyxy.tolist()
            for tup in zip(res_boxes):
                box = tup
                object = [int(box[0]), int(box[1]), int(box[2]- box[0]), int(box[3] - box[1])]
                x,y,w,h = object
                plate_detection(image[y:y+h, x:x+w])
                break
        if n > 80:
            break
    end = time.time()
    print(end - start)
if __name__ == "__main__":

    # i = 0
    # camera.capture("/home/zhaklee/Desktop/thesis/baseline.jpg")
    # Load a pretrained YOLOv8n model
    # source = "/home/zhaklee/Desktop/thesis/baseline.jpg"
    capture = cv2.VideoCapture("utube.mp4")
    success,image = capture.read()
    # Run inference on the source
    results = model(image, save=True, classes=[3,4,6])  # list of Results objects
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()
    masks = results[0].masks

    h, w = results[0].orig_shape



    for i, tup in enumerate(zip(boxes, classes, confidences, masks)):
        _, cls, _, mask = tup

        minX.append(9999)
        minY.append(9999)
        maxX.append(0)
        maxY.append(0)
        for point in mask.xy[0]:
            if minX[i] > point[0]:
                minX[i] = point[0]
            if minY[i] > point[1]:
                minY[i] = point[1]
            if maxX[i] < point[0]:
                maxX[i] = point[0]
            if maxY[i] < point[1]:
                maxY[i] = point[1]
        points_list.append(mask.xy[0])
    process_video()
    # while True:
    #     button.when_pressed = process_video
    #     sleep(5)
    # button.close()

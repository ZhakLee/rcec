from ultralytics import YOLO
from time import sleep
import time
import numpy as np
import cv2
name = "utube.mp4"
vid = "1"
run = 1

model = YOLO("/home/zhaklee/Desktop/Thesis/Thesis/weights/road.pt")

points_list = []
capture = cv2.VideoCapture(name)
success, frame = capture.read()

minX = []
minY = []
maxX = []
maxY = []
count = 0
plate = YOLO("/home/zhaklee/Desktop/Thesis/Thesis/weights/plate.pt")
vehicle = YOLO("yolov8n-seg.pt")


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

def plate_detection(img, i):
    global plate
    global count
    global run
    global vid
    results = plate(img, save=True) 
    boxes  = results[0].boxes.xyxy.tolist()
    confs  = results[0].boxes.conf.tolist()
    for box, conf in zip(boxes,confs):
        object = [int(box[0]), int(box[1]), int(box[2]- box[0]), int(box[3] - box[1])]
        x,y,w,h = object   
        cv2.imwrite(vid + "_" + str(run) +"_vehicle_" + i +'_detected_plate_' + str(count) +  "_conf_" + str(conf) + '.jpg', img[y:y+h, x:x+w])
        count += 1

def process_video():
    global model
    # global capture
    global success
    global run
    global vid
    n = 0
    # camera.start_recording("/home/zhaklee/Desktop/thesis/redlight.h264")
    capture = cv2.VideoCapture("utube.mp4")
    success,image = capture.read()
    start = time.time()
    i = 0
    while success:
        success,image = capture.read()
        n += 1
        if n > (run-1)*150 and n % 20 == 0:
            result = vehicle(image, save=True, classes = [2,3,5,7])
            res_masks = result[0].masks
            res_boxes = result[0].boxes.xyxy.tolist()
            res_conf  = result[0].boxes.conf.tolist()
            print(res_conf)
            for tup in zip(res_boxes,res_masks,res_conf):
                box, mask, conf = tup
                points = (mask.xy[0])
                for point in points:
                    # print (point[0], point[1])
                    if detect_collision(point[0], point[1]) > 0:
                        i += 1
                        object = [int(box[0]), int(box[1]), int(box[2]- box[0]), int(box[3] - box[1])]
                        x,y,w,h = object
                        cv2.imwrite(vid + "_" + str(run) +"_vehicle_" + str(i) + "_with_conf_" + str(conf)+".jpg", image[y:y+h, x:x+w])
                        plate_detection(image[y:y+h, x:x+w], str(i))
                        break
        if n > 150*run:
            break
    end = time.time()
    print(end - start)
if __name__ == "__main__":

    # i = 0
    # camera.capture("/home/zhaklee/Desktop/thesis/baseline.jpg")
    # Load a pretrained YOLOv8n model
    # source = "/home/zhaklee/Desktop/thesis/baseline.jpg"
    capture = cv2.VideoCapture(name)
    success,image = capture.read()
    # Run inference on the source
    results = model(image, save=True, classes = [4,11])  # list of Results objects
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

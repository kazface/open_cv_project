import time

import cv2 as cv2
import numpy as np
from datetime import datetime
import webcolors


mouse_x = mouse_y = 0
r = g = b = 0
clicked = False
clicked_time = datetime.now()


def detectColor(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        global r, g, b, clicked, mouse_x, mouse_y
        b, g, r = frame[y, x]

        b, g, r = int(b), int(g), int(r)

        mouse_x, mouse_y = x, y

        clicked = True




def drawShape(frame, x, y, w, h, text):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


# filter contours by area > 10_000 and then draw it
def drawSummary(frame, text):
    cv2.putText(frame, text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


def detectShapes(frame,contours):
    _triangle_count = 0
    _total_count = 0

    for contour in contours:  # check area of counter
        area = cv2.contourArea(contour)
        if area > 5_000:
            cv2.drawContours(frame, contour, -1, (200,200,0),3)
            p = cv2.arcLength(contour, True)
            vert_count = cv2.approxPolyDP(contour, 0.01 * p, True)
            x, y, w, h = cv2.boundingRect(vert_count)
            if len(vert_count) == 3:
                drawShape(frame, x, y, w, h, "Triangle")
                _triangle_count += 1
            elif len(vert_count) == 4:
                drawShape(frame, x, y, w, h, "Rectangle")
            elif len(vert_count) == 5:
                drawShape(frame, x, y, w, h, "Pentagon")
            elif len(vert_count) == 6:
                drawShape(frame, x, y, w, h, "Hexagon")
            elif len(vert_count) == 7:
                drawShape(frame, x, y, w, h, "Heptagon")
            elif len(vert_count) == 8:
                drawShape(frame, x, y, w, h, "Octagon")
            _total_count += 1

    drawSummary(frame, f'''Total object count: {_total_count}; Triangles: {_triangle_count}''')




def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def get_colour_name(requested_colour):
    try:
        closest_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
    return closest_name


def detect_Object():
    Live_cam = cv.VideoCapture(0)
    Live_cam.set(3,1280)
    Live_cam.set(4,720)
    Live_cam.set(10,70)
    threshold = 0.5
    objectname= []
    cocofile = 'largest\\coco.names'
    with open(cocofile,'rt') as f:
        objectname = f.read().rstrip('\n').split('\n')

    configPath = 'largest\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'largest\\frozen_inference_graph.pb'

    net = cv.dnn_DetectionModel(weightsPath,configPath)
    net.setInputSize(320,320)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    while True:
        success,img = Live_cam.read()
        objectID, confs, bbox = net.detect(img,confThreshold=threshold)
        if len(objectID) != 0:
            for classId, confidence,box in zip(objectID.flatten(),confs.flatten(),bbox):
                #Add the rectangle to the object the is detect
                cv.rectangle(img,box,color=(0,255,0),thickness=2)
                #Add the object name 
                cv.putText(img,objectname[classId-1].upper(),(box[0]+10,box[1]+30),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),cv.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv.imshow("Output",img)
        cv.waitKey(1)



if __name__ == '__main__':

    capture = cv2.VideoCapture("2D Shapes for Kids.mp4") #webcam live video
    cv2.namedWindow("track")
    cv2.createTrackbar("T1", "track", 0, 255, lambda x: x)
    cv2.createTrackbar("T2", "track", 0, 255, lambda x: x)
    kernel = np.ones((5, 5))
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))

    while True:
        ret, frame = capture.read()
        frame = cv2.bilateralFilter(frame, 9, 75, 75)
        lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        _l, _a, _b = cv2.split(lab)
        l2 = clahe.apply(_l)
        lab = cv2.merge((l2, _a, _b))

        contrast = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        gray = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)

        thresh1 = cv2.getTrackbarPos("T1", "track")
        thresh2 = cv2.getTrackbarPos("T2", "track")
        canny = cv2.Canny(gray, thresh1, thresh2)
        dil = cv2.dilate(canny, kernel, iterations = 1)

        contours, h = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if(clicked):
            clicked_time = datetime.now()
            # cv2.putText(frame, f"R = {r}, G={g}, B={b}", (mouse_x, mouse_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            clicked = False

        if (datetime.now() - clicked_time).total_seconds() < 2:
            cv2.putText(frame, f"color: {get_colour_name((r, g, b))}", (mouse_x, mouse_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)


        detectShapes(frame, contours)

        cv2.imshow("frame", frame)
        cv2.imshow("dil", dil)
        cv2.imshow("canny", canny)

        cv2.setMouseCallback("frame", detectColor)




        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

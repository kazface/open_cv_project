import cv2
import numpy as np
from datetime import datetime
import webcolors
from math import isclose


mouse_x = mouse_y = 0
r = g = b = 0
clicked = False
clicked_time = datetime.now()

def resize(frame):
    weidth = int(frame.shape[0] * 1)
    height = int(frame.shape[1] * .35)
    dimensions = (weidth,height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def detect_color(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        global r, g, b, clicked, mouse_x, mouse_y
        b, g, r = frame[y, x]
        b, g, r = int(b), int(g), int(r)
        mouse_x, mouse_y = x, y
        clicked = True

def draw_shape_bottom(frame, x, y, w, h, text, margin_bottom):
    cv2.putText(frame, text, (x+margin_bottom, y+margin_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 150, 12), 2)


def draw_shape(frame, x, y, w, h, text):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


def detect_colors(frame, contours, orig_frame):
    for contour in contours:  # check area of counter'
        area = cv2.contourArea(contour)
        if(area > 5_000):
            x, y, w, h = cv2.boundingRect(contour)
            b,g,r = np.mean(orig_frame[y:y + h, x:x + w], axis=(0,1))
            draw_shape_bottom(frame, x, y, w, h, f"Average color: {get_colour_name((r,g,b))}", h//2+25)
    pass


# filter contours by area > 10_000 and then draw it
def detect_shapes(frame, contours):
    _triangle_count = 0
    _total_count = 0
    _biggest_shape = {
        "area": 0,
        "params": {0, 0, 0, 0} #x, y, w, h
    }
    _smallest_shape = {
        "area": np.inf,
        "params": {0, 0, 0, 0} #x, y, w, h
    }


    for contour in contours:  # check area of counter
        area = cv2.contourArea(contour)
        if area > 5_000:
            cv2.drawContours(frame, contour, -1, (200,200,0),3)
            p = cv2.arcLength(contour, True)
            vert_count = cv2.approxPolyDP(contour, 0.01 * p, True)
            x, y, w, h = cv2.boundingRect(vert_count)

            if area > _biggest_shape['area']:
                _biggest_shape['area'] = area
                _biggest_shape['params'] = x, y, w, h

            if area < _smallest_shape['area']:
                _smallest_shape['area'] = area
                _smallest_shape['params'] = x, y, w, h

            if len(vert_count) == 3:
                draw_shape(frame, x, y, w, h, "Triangle")
                _triangle_count += 1
            elif len(vert_count) == 4:
                draw_shape(frame, x, y, w, h, "Rectangle")
            elif len(vert_count) == 5:
                draw_shape(frame, x, y, w, h, "Pentagon")
            elif len(vert_count) == 6:
                draw_shape(frame, x, y, w, h, "Hexagon")
            elif len(vert_count) == 7:
                draw_shape(frame, x, y, w, h, "Heptagon")
            elif len(vert_count) == 8:
                draw_shape(frame, x, y, w, h, "Octagon")
            _total_count += 1
    try:
        if isclose(_biggest_shape['area'], _smallest_shape['area'], abs_tol=100):
            draw_shape_bottom(frame, *_biggest_shape['params'], "Biggest Shape", margin_bottom=25)
            draw_shape_bottom(frame, *_biggest_shape['params'], "Smallest Shape", margin_bottom=65)
        else:
            draw_shape_bottom(frame, *_biggest_shape['params'], "Biggest Shape", margin_bottom=25)
            draw_shape_bottom(frame, *_smallest_shape['params'], "Smallest Shape", margin_bottom=25)
    except:
        pass
    cv2.putText(frame, f"Total object count: {_total_count}", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
    cv2.putText(frame, f"Triangles: {_triangle_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)



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
    Live_cam = cv2.VideoCapture(0)
    Live_cam.set(3,1280)
    Live_cam.set(4,720)
    Live_cam.set(10,70)
    threshold = 0.45
    objectname= []
    cocofile = 'largest\\coco.names'
    with open(cocofile,'rt') as f:
        objectname = f.read().rstrip('\n').split('\n')

    configPath = 'largest\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'largest\\frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath,configPath)
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
                cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                #Add the object name
                cv2.putText(img,objectname[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow("Output",resize(img))
        cv2.waitKey(1)



if __name__ == '__main__':
    capture = cv2.VideoCapture("Baby Sensory - Shapes & Transitions with Deep Relaxation Ambient Music & Sound Effects - Baby Sleep.mp4") #webcam live video
    cv2.namedWindow("threshold")
    cv2.createTrackbar("T1", "threshold", 0, 255, lambda x: x)
    cv2.createTrackbar("T2", "threshold", 0, 255, lambda x: x)

    cv2.namedWindow("color")
    cv2.createTrackbar("H", "color", 0, 180, lambda x: x)
    cv2.createTrackbar("S", "color", 0, 255, lambda x: x)
    cv2.createTrackbar("V", "color", 0, 255, lambda x: x)
    cv2.createTrackbar("HL", "color", 0, 180, lambda x: x)
    cv2.createTrackbar("SL", "color", 0, 255, lambda x: x)
    cv2.createTrackbar("VL", "color", 0, 255, lambda x: x)


    kernel = np.ones((5, 5))

    #Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
    while True:

        ret, frame = capture.read()
        frame = cv2.bilateralFilter(frame, 9, 75, 75)
        frame_orig = frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lab  = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        _l, _a, _b = cv2.split(lab) # split on 3 different channels

        l2 = clahe.apply(_l) #apply CLAHE to the L-channel
        lab = cv2.merge((l2, _a, _b)) # merge channels
        contrast = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        gray = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)

        thresh1 = cv2.getTrackbarPos("T1", "threshold")
        thresh2 = cv2.getTrackbarPos("T2", "threshold")

        h_track = cv2.getTrackbarPos("H", "color")
        s_track = cv2.getTrackbarPos("S", "color")
        v_track = cv2.getTrackbarPos("V", "color")
        hl_track = cv2.getTrackbarPos("HL", "color")
        sl_track = cv2.getTrackbarPos("SL", "color")
        vl_track = cv2.getTrackbarPos("VL", "color")

        lower = np.array([hl_track, sl_track, vl_track]) #lower tr for color detection
        upper = np.array([h_track, s_track, v_track]) #upper tr for color detection
        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN,kernel)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,kernel)

        canny = cv2.Canny(gray, thresh1, thresh2)

        dil = cv2.dilate(canny, kernel, iterations = 1)

        contours, h = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        contours = sorted(contours, key = cv2.contourArea, reverse= True) #sort from biggest to smallest

        contours_color, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours_color = sorted(contours_color, key = cv2.contourArea, reverse= True) #sort from biggest to smallest

        if(clicked):
            clicked_time = datetime.now()
            # cv2.putText(frame, f"R = {r}, G={g}, B={b}", (mouse_x, mouse_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            clicked = False
        if (datetime.now() - clicked_time).total_seconds() < 2:
            cv2.putText(frame, f"color: {get_colour_name((r, g, b))}", (mouse_x, mouse_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)

        detect_shapes(frame, contours)
        detect_colors(frame, contours_color, frame_orig)
        cv2.imshow("mask", mask)
        cv2.imshow("frame", frame)
        cv2.imshow("dil", dil)
        cv2.imshow("canny", canny)
        cv2.setMouseCallback("frame", detect_color)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()

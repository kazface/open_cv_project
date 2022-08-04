import cv2 as cv2
import numpy as np


def draw(frame, x, y, w, h, text):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


if __name__ == '__main__':

    capture = cv2.VideoCapture(0) #webcam live video


    cv2.namedWindow("track")


    cv2.createTrackbar("T1", "track", 0, 255, lambda x: x)

    cv2.createTrackbar("T2", "track", 0, 255, lambda x: x)
    kernel = np.ones((5, 5))

    while True:
        ret, frame = capture.read() #
        # frame = cv2.GaussianBlur(frame, (5, 5), 9) corners are blurred (not use)
        frame = cv2.bilateralFilter(frame, 9, 75, 75)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



        thresh1 = cv2.getTrackbarPos("T1", "track")
        thresh2 = cv2.getTrackbarPos("T2", "track")
        canny = cv2.Canny(gray, thresh1, thresh2)

        dil = cv2.dilate(canny, kernel, iterations=1)


        contours, h = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #filter contours by area > 10_000 and then draw it

        for contour in contours: #check area of counter
            area = cv2.contourArea(contour)
            if area > 5_000:
                cv2.drawContours(frame, contour, -1, (200, 200, 0), 2)
                p = cv2.arcLength(contour, True)
                vert_count = cv2.approxPolyDP(contour, 0.01 * p, True)
                x, y, w, h = cv2.boundingRect(vert_count)

                if(len(vert_count) == 3):
                    draw(frame, x, y, w, h,"Triangle")
                elif(len(vert_count) == 4):
                    draw(frame, x, y, w, h,"Rectangle")
                elif(len(vert_count) == 5):
                    draw(frame, x, y, w, h,"Pentagon")
                elif(len(vert_count) == 10):
                    draw(frame, x, y, w, h,"Star")
                else:
                    draw(frame, x, y, w, h , "Shape")


        # [cv2.drawContours(frame, contour, -1, (200, 200, 0), 3) for contour in contours if cv2.contourArea(contour) > 10_000]



        cv2.imshow("frame", frame)
        cv2.imshow("dil", dil)
        cv2.imshow("canny", canny)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
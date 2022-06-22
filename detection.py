#!/usr/bin/env python3

import cv2
from cv2 import *
import numpy as np
from scipy.stats import itemfreq
import rospy 
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3Stamped

def publisher(val):
    global obj
    obj = Twist()
    val = float(val)

    rospy.init_node('traffic_signs')

    obj.linear.x = val
    obj.linear.y = 0.0
    obj.linear.z = 0.0
    obj.angular.x = 0.0
    obj.angular.y = 0.0
    obj.angular.z = 0.0

        
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=50)
        
    pub.publish(obj)

def get_dominant_color(image, n_colors):
    pixels = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    flags, labels, centroids = cv2.kmeans(
        pixels, n_colors, None, criteria, 10, flags)
    palette = np.uint8(centroids)
    return palette[np.argmax(itemfreq(labels)[:, -1])]


clicked = False
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True



def funct():
    cameraCapture = cv2.VideoCapture(1) 
    #cv2.namedWindow('camera')
    #cv2.setMouseCallback('camera', onMouse)

    # # Read and process frames in loop
    success, frame = cameraCapture.read()
    # cv2.imshow('Zone2', frame)


    c = 0
    while success and not clicked:
        c = cv2.waitKey(1)
        if c == 'q':
            break 
        
        success, frame = cameraCapture.read()
        
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(gray, 37)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 50, param1=120, param2=40)

        if not circles is None:
            circles = np.uint16(np.around(circles))
            max_r, max_i = 0, 0
            for i in range(len(circles[:, :, 2][0])):
                if circles[:, :, 2][0][i] > 50 and circles[:, :, 2][0][i] > max_r:
                    max_i = i
                    max_r = circles[:, :, 2][0][i]
            x, y, r = circles[:, :, :][0][max_i]
            if y > r and x > r:
                square = frame[y-r:y+r, x-r:x+r]
                
                dominant_color = get_dominant_color(square, 2)
                # print("domCOL_0", dominant_color[0])
                # print("domCOL_1", dominant_color[1])
                # print("domCOL_2", dominant_color[2])
                 
                if dominant_color[0] < 50:
                    if dominant_color[2] < 50:
                        val = 0.9
			        print("forward")
                    else:
                        val = 0.0
                        print("STOP")

                    
                elif dominant_color[0] > 111:
                    if dominant_color[2] < 180:
                        val = 0.6
                        print("30")
                    if dominant_color[2] > 200:
                        val = 0.7
                        print("Traffic light - GREEN GO")
    
                else:
                    print("N/A")

            publisher(val)

            for i in circles[0, :]:
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
    #   cv2.imshow('camera', frame)
        


if __name__ == '__main__':
    try:
        funct()
    except rospy.ROSInterruptException:
        pass

cv2.destroyAllWindows()
cameraCapture.release()

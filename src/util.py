import signal
import sys

import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError


def interuppt_handler(signum, frame):
    print("Signal handler!!!")
    sys.exit(-2) #Terminate process here as catching the signal removes the close process behaviour of Ctrl-C


def _process_image(img, show_image=True):
    bridge = CvBridge()
    image = bridge.imgmsg_to_cv2(img, "bgr8")
    gray = bridge.imgmsg_to_cv2(img, "mono8")
    _, gray = cv.threshold(gray, 160, 255, cv.THRESH_BINARY)

    H, W, _ = image.shape
    clipped = gray[int(H*2/3):, :]

    cnt, _, _, centroids = cv.connectedComponentsWithStats(clipped)

    if cnt < 3:
        return None

    distances = [x[0] - W/2 for x in centroids]
    left_edge = centroids[distances.index(min(distances))]
    right_edge = centroids[distances.index(max(distances))]

    # print("EDGES:", left_edge, right_edge)

    if len(left_edge) == 0 or len(right_edge) == 0 or (left_edge == right_edge).all():
        return None

    if left_edge[0] < 30 or right_edge[0] > W - 30:
        return None

    goal_x = int((left_edge[0] + right_edge[0]) / 2)
    goal_y = int((left_edge[1] + right_edge[1]) / 2)
    error = goal_x - W / 2

    slope = error * (180.0 / W)

    clipped = cv.cvtColor(clipped, cv.COLOR_GRAY2RGB)
    pt0 = (int(W/2), int(H/3))
    pt1 = (goal_x, goal_y)
    pt2 = (int(W/2), goal_y)

    cv.arrowedLine(clipped, pt0, pt1, (0,255,0), 2, 8, 0, 0.03)
    cv.arrowedLine(clipped, pt0, pt2, (255,0,0), 2, 8, 0, 0.03)

    cv.imwrite("clipped.png", clipped)
    if show_image:
        cv.imshow("clipped", clipped)
        cv.waitKey(1)

    return slope/90.0

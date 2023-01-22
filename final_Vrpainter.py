import cv2
import numpy as np
import mediapipe as mp
import math
# from cvzone.SelfiSegmentationModule import SelfiSegmentation
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
list_line1 = []
list_line2 = []
imgCanvas = np.zeros((720, 1280, 3),np.uint8)
# segmentor = SelfiSegmentation()
def get_pose(img_rgb, draw):
    results = pose.process(img_rgb)
    if draw :
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x *w), int(lm.y * h)
                cv2.putText(img, str(id), (cx, cy),cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    else:
        return img, results
    return img, results

def findPose(img, results):
    listPose = []
    h, w, _ = img.shape
    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, _ = img.shape
            cx, cy = int(lm.x *w), int(lm.y * h)
            listPose.append([id, cx, cy])
            
    return listPose

# def findHand(listPose):
#     x19, y19 = listPose[19][1], listPose[19][2]
#     x20, y20 = listPose[20][1], listPose[20][2]
#     print(x19, y19, x20, y20)
    # print(x20, y20)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    cap.set(10, 0)
    while True:
        ret, img = cap.read()
        listPose = []
        idx_pose = []
        try:
            if ret:
                img = cv2.flip(img, 1)
                bg = np.zeros_like(img)
                img =  cv2.addWeighted(img, 0.5, bg, 0.5, 0)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
               
                mask = np.zeros_like(img)
                img, results = get_pose(img_rgb, draw=False)
                listPose = findPose(img, results)
                if len(listPose) != 0:
                    x1, y1 = listPose[19][1], listPose[19][2]
                    x2, y2 = listPose[20][1], listPose[20][2]
                    
                    list_line1.insert(0, [x1, y1])
                    list_line2.insert(0, [x2, y2])
                    k = len(list_line1) - 2 
                    k2 = len(list_line2) - 2
                    if k > 10:
                        k = 10    
                    
                        # cv2.circle(img, (list_line[n][0], list_line[n][1]), 6, (150,0,150), cv2.FILLED)
                        # cv2.circle(img, (list_line[n][0], list_line[n][1]), 5, (200,0,200), cv2.FILLED)
                        # cv2.circle(img, (list_line[n][0], list_line[n][1]), 3, drawColor, cv2.FILLED)
                        
                    t = 255
                    r = 255    
                    for i in range (1, 11):
                        n = k
                        while n > 0:
                            cv2.line(img, (list_line1[n - 1][0], list_line1[n - 1][1]), (list_line1[n][0], list_line1[n][1]),
                                     (t,0, r), 24- 2*i)
                            cv2.line(img, (list_line2[n - 1][0], list_line2[n - 1][1]), (list_line2[n][0], list_line2[n][1]),
                                     (t,0, r), 24- 2*i)
                            # cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                            n -= 1
                        t -= 4*i
                        r -= 3
                    n = k
                    while n > 0:    
                            cv2.line(img, (list_line1[n - 1][0], list_line1[n - 1][1]), (list_line1[n][0], list_line1[n][1]),
                                     (200, 200, 200), 4)
                            cv2.line(img, (list_line2[n - 1][0], list_line2[n - 1][1]), (list_line2[n][0], list_line2[n][1]),
                                     (200, 200, 200), 4)
                            # cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                            n -= 1
                gray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
                _, imginv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
                
                imginv = cv2.cvtColor(imginv, cv2.COLOR_GRAY2BGR)
                img = cv2.bitwise_and(img, imginv)
                img = cv2.bitwise_or(img, imgCanvas)
                
                cv2.imshow('img',img)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
            else:
                break
        except :
            print("_____end_______")  
    cap.release()
    cv2.destroyAllWindows()
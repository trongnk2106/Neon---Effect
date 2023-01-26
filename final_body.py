import cv2
import numpy as np
import mediapipe as mp
import math
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
segmentor = SelfiSegmentation()

def remove_bg(img):
    out = segmentor.removeBG(img, (0, 0, 0))
    return out

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

def findDistancehead(listPose):
    x22, y22 = listPose[22][1], listPose[22][2]
    x21, y21 = listPose[21][1], listPose[21][2]
    x12, y12 = listPose[12][1], listPose[12][2]
    x11, y11 = listPose[11][1], listPose[11][2]
    x7, y7 = abs(listPose[7][1]), abs(listPose[7][2])
    x8, y8 = abs(listPose[8][1]), abs(listPose[8][2])
    dist228= int(math.hypot(x8 - x22, y8 - y22))
    dist128 = int(math.hypot(x8 - x12, y8 - y12))
    dist217 = int(math.hypot( x7- x21, y7 - y21))
    dist117 = int(math.hypot(x7 - x11, y7 - y11))
    
    if dist228 < dist128 or dist217 < dist117:
        return True
    return False

def findDistancebody(listPose):
    x16, y16 = listPose[16][1], listPose[16][2]
    x15, y15 = listPose[15][1], listPose[15][2]
    x11, y11 = listPose[11][1], listPose[11][2]
    x12, y12 = listPose[12][1], listPose[12][2]
    dist1611 = math.hypot(x16 - x11 , y16 - y11)
    dist1512 = math.hypot(x15 - x12, y15 - y12)
    dist1211 = math.hypot(x12-x11, y12-y11)
    if dist1512 < dist1211 or dist1611 < dist1211 :
        return True
    return False

def findDistanceleg(listPose):
    x16, y16 = listPose[16][1], listPose[16][2]
    x23, y23 = listPose[23][1], listPose[23][2]
    x15, y15 = listPose[15][1], listPose[15][2]
    x24, y24 = listPose[24][1], listPose[24][2]
    x12, y12 = listPose[12][1], listPose[12][2]
    dist1623 = math.hypot(x16 - x23, y16 - y23)
    dist1524 = math.hypot(x15 - x24, y15 - y24)
    dist2324 = math.hypot(x23- x24, y23 -y24)
    if  (dist1623 < dist2324 or dist1524 < dist2324) :
        return True
    return False

def points2hand(listPose):
    x14, y14 = listPose[14][1], listPose[14][2]
    x12, y12 = listPose[12][1], listPose[12][2]
    x11, y11 = listPose[11][1], listPose[11][2]
    x13, y13 = listPose[13][1], listPose[13][2]
    if y14 < y12 and y13 < y11:
        return True
    return False

def find2hand(listPose):
    x16, y16 = listPose[16][1], listPose[16][2]
    x15, y15 = listPose[15][1], listPose[15][2]
    x11, y11 = listPose[11][1], listPose[11][2]
    x12, y12 = listPose[12][1], listPose[12][2]

    d = abs((x11 - x12)) //2
    dist156 = math.hypot(x16 - x15, y16 -y15)
    if dist156 < d :
        return True
    return False

def liftleg(listPose):
    x26, y26 = listPose[26][1], listPose[26][2]
    x25, y25 = listPose[25][1], listPose[25][2]
    x24, y24 = listPose[24][1], listPose[24][2]
    x23, y23 = listPose[23][1], listPose[23][2]
    
    d1 = abs((y25 - y24))//4
    d2 = abs((y26 - y23))//4
    
    if y25 < y26-d2 :
        return 2
    if y26 < y25 - d1:
        return 1
    return 0

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
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
                res = remove_bg(img)
                gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
                img_blur = cv2.GaussianBlur(gray, (7, 7), 0)
                edges = cv2.Canny(img_blur, threshold1 = 0, threshold2 = 200)
                idx = np.array(np.squeeze(cv2.findNonZero(edges)))
                edges_ = cv2.Canny(img_blur, threshold1 = 0, threshold2 = 200)
                edges_ = np.dstack((edges, edges_, edges_))
                mask = np.zeros_like(img)
                img, results = get_pose(img_rgb, draw=False)
                listPose = findPose(img, results)
                if len(listPose):
                    if findDistancehead(listPose) == True:
                        for te, i in enumerate(idx):
                            if listPose[12][1] < i[0] < listPose[11][1] and i[1] < listPose[12][2]:
                                idx_pose.append(i)    
                    if findDistancebody(listPose) == True: 
                        d = (listPose[12][2] - listPose[10][2]) //2
                        for tr, j in enumerate(idx):
                            if listPose[12][2] - d <= j[1] <= listPose[24][2] :
                                idx_pose.append(j)
                    if findDistanceleg(listPose) == True:
                        for e, l in enumerate(idx):
                            if listPose[24][2] + 10 < l[1] < listPose[28][2]:
                                idx_pose.append(l)
                    if find2hand(listPose) == True:
                        for te, i in enumerate(idx):
                            idx_pose.append(i)
                    if liftleg(listPose) == 1:
                        
                        x24, y24 = listPose[24][1], listPose[24][2]
                        x23, y23 = listPose[23][1], listPose[23][2]
                        mid = abs(x23 + x24) // 2
                        for e, l in enumerate(idx):
                            if l[1] > y24 and l[0] < mid: 
                                # print("in for")
                                idx_pose.append(l)
                    if liftleg(listPose) == 2:
                       
                        x24, y24 = listPose[24][1], listPose[24][2]
                        x23, y23 = listPose[23][1], listPose[23][2]
                        mid = abs(x23 + x24) // 2
                        
                        for e, l in enumerate(idx):
                            if l[1] > y23 and l[0] > mid: 
                                idx_pose.append(l)
                                
                    if len(idx_pose) != 0 : 
                        for i in idx_pose:
                            # img[i[1] : i[1] + 5, i[0] : i[0] + 5, 0] = 255
                            # img[i[1] : i[1] + 5, i[0] : i[0] + 5, 1] = 0
                            # img[i[1] : i[1] + 5, i[0] : i[0] + 5, 2] = 255
                            
                            
                            edges_[i[1] - 1: i[1] + 1, i[0] - 1: i[0]+ 1, 0] = 255
                            edges_[i[1] -1 : i[1] + 1, i[0] - 1 : i[0] + 1, 1] = 255
                            edges_[i[1] - 1: i[1] + 1, i[0] - 1 : i[0] + 1, 2] = 255
                            
                            edges_[i[1] + 2: i[1] + 4, i[0] + 2 : i[0]+ 4, 0] = 255
                            edges_[i[1] + 2 : i[1] + 4, i[0] + 2: i[0] + 4, 1] = 0
                            edges_[i[1] + 2: i[1] + 4, i[0] + 2: i[0] + 4, 2] = 255
                            
                            edges_[i[1] - 2: i[1] - 4, i[0]  - 2: i[0] - 4, 0] = 255
                            edges_[i[1] - 2 : i[1] -4, i[0] - 2: i[0]  - 4, 1] = 0
                            edges_[i[1]  - 2: i[1] - 4, i[0] - 2: i[0] - 4, 2] = 255
                            
                            # mask[i[1], i[0] - 1: i[0]+ 1, 0] = 255
                            # mask[i[1], i[0] - 1 : i[0] + 1, 1] = 255
                            # mask[i[1], i[0] - 1 : i[0] + 1, 2] = 255
                            
                            # mask[i[1] , i[0] + 2 : i[0]+ 4, 0] = 255
                            # mask[i[1] , i[0] + 2: i[0] + 4, 1] = 0
                            # mask[i[1], i[0] + 2: i[0] + 4, 2] = 255
                            
                            # mask[i[1], i[0]  - 2: i[0] - 4, 0] = 255
                            # mask[i[1], i[0] - 2: i[0]  - 4, 1] = 0
                            # mask[i[1] , i[0] - 2: i[0] - 4, 2] = 255
                            
                            
                            
                            
                            
                            
                            
                            # mask[i[1] - 1: i[1] + 1, i[0] - 1: i[0]+ 1, 0] = 255
                            # mask[i[1] -1 : i[1] + 1, i[0] - 1 : i[0] + 1, 1] = 255
                            # mask[i[1] - 1: i[1] + 1, i[0] - 1 : i[0] + 1, 2] = 255
                            
                            # mask[i[1] + 2: i[1] + 4, i[0] + 2 : i[0]+ 4, 0] = 255
                            # mask[i[1] + 2 : i[1] + 4, i[0] + 2: i[0] + 4, 1] = 0
                            # mask[i[1] + 2: i[1] + 4, i[0] + 2: i[0] + 4, 2] = 255
                            
                            # mask[i[1] - 2: i[1] - 4, i[0]  - 2: i[0] - 4, 0] = 255
                            # mask[i[1] - 2 : i[1] -4, i[0] - 2: i[0]  - 4, 1] = 0
                            # mask[i[1]  - 2: i[1] - 4, i[0] - 2: i[0] - 4, 2] = 255
                else:
                    idx_pose = []
                cv2.imshow('img',edges_)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
            else:
                break
        except :
            print("_____end_______")  
    cap.release()
    cv2.destroyAllWindows()
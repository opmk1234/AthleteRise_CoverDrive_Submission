import cv2
import mediapipe as mp
from matplotlib import pyplot as plt
import os
import time

class PoseDetector():
 def __init__ (self,mode= False,upBody=False,smooth= True,DetectCon=0.5,TrackCon=0.5):
  self.mode   = mode
  self.upBody = upBody
  self.smooth = smooth
  self.DetectCon = DetectCon
  self.TrackCon= TrackCon 
  self.mpDrawing = mp.solutions.drawing_utils
  self.mpPose = mp.solutions.pose
  self.pose = self.mpPose.Pose(
  static_image_mode=self.mode,
  
  smooth_landmarks=self.smooth,
  min_detection_confidence=self.DetectCon,
  min_tracking_confidence=self.TrackCon)

 def findpose(self,frame,draw=True):
  self.rgbframe = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
  self.result =self.pose.process(self.rgbframe)
  if self.result.pose_landmarks:
    if draw:
     self.mpDrawing.draw_landmarks(frame,self.result.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
  return frame 
 
 def findposition(self,frame,draw=True):
    lmList=[]
    if self.result.pose_landmarks:
     for id,lm in enumerate(self.result.pose_landmarks.landmark):
      h,w,c= frame.shape
      cx,cy =int(lm.x*w),int(lm.y*h)
      lmList.append([id,cx,cy])
      if draw:
       cv2.circle(frame,(cx,cy),5,(255,0,0),cv2.FILLED)
    return lmList


def main():
 cap =cv2.VideoCapture(os.path.join('Poses','Pose2.mp4'))
 detector = PoseDetector()
 ptime= 0
 while (True):
  succ,frame = cap.read()
  frame= detector.findpose(frame)
  lmList=detector.findposition(frame,draw=False)
  print(lmList[14])
  cv2.circle(frame,(lmList[14][1],lmList[14][2]),15,(0,0,255),cv2.FILLED)
    
  ctime = time.time()
  if  ctime != 0:
   fps = 1/(ctime - ptime)
   ptime = ctime
   cv2.putText(frame,str(int(fps)),(70,50),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,0),3)
  cv2.namedWindow("image", cv2.WINDOW_NORMAL)
  cv2.imshow("image",frame)
  cv2.waitKey(1)


 



if __name__ == "__main__":
   main()
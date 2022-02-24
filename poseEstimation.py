import mediapipe as mp
import cv2 as cv

class poseDetector():
    def __init__(self, mode = False, upper_body_only = False, smooth_landmarks = True,
                 min_detection_confidence = 0.5, min_tracking_confidence = 0.5):
        self.mode = mode
        self.upper_body_only = upper_body_only
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()#self.mode,self.upper_body_only,self.smooth_landmarks,
                                     #self.min_detection_confidence, 
                                    # self.min_tracking_confidence)
        
        
        
        
    def findPoseEstimation(self, img, draw = True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        
        if self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    
    def findPositions(self, img, draw = True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                    cv.circle(img, (cx,cy), 6, (255,0,0),cv.FILLED)
        return lmList
    
    def displayPoseEstimation():
        cap = cv.VideoCapture('ExampleVideos/napolean_dance.mp4')
        detector = poseDetector()
        
        while True:
            success, img = cap.read()
            if success:
                img = cv.resize(img, (750,500))
                img = detector.findPoseEstimation(img)
                lmList = detector.findPositions(img, draw = False)#set to false bc drawing 
                #elbow position below 
                
                #track elbow ex below. refer to pose_landmarks in reference pictures
                if len(lmList) != 0:
                    cv.circle(img, (lmList[14][1],lmList[12][2]), 10, (0,0,255),cv.FILLED)
                
                
                cv.imshow("Image", img)
                k = cv.waitKey(1) & 0xFF
                if k == 27:
                    break
            else:
                print("No VideoCapture Detected :C")
                break;
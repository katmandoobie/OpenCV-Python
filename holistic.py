from turtle import Turtle
import mediapipe as mp
import cv2 as cv

class holisticDetector():
    def __init__(self, mode = False, upper_body_only = False, smooth_landmarks = True,
                 min_detection_confidence = 0.5, min_tracking_confidence = 0.5):
        self.mode = mode
        self.upper_body_only = upper_body_only
        self.smooth_landmarks = smooth_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
    
        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawStyles = mp.solutions.drawing_styles
        self.mpHolistic = mp.solutions.holistic
        self.holistic = self.mpHolistic.Holistic()
       
        
        
        
    def findHolistics(self, img):
        #to improve performace set img as not writeable
        img.flags.writeable = False
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.holistic.process(img)
        
        img.flags.writeable = True
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    
        self.mpDraw.draw_landmarks(img,self.results.face_landmarks,self.mpHolistic.FACEMESH_CONTOURS,landmark_drawing_spec=None,connection_drawing_spec=self.mpDrawStyles.get_default_face_mesh_contours_style())
        self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpHolistic.POSE_CONNECTIONS,landmark_drawing_spec= self.mpDrawStyles.get_default_pose_landmarks_style())
        return img
    
    
    
    def displayHolistics():
        cap = cv.VideoCapture(0)
        detector = holisticDetector()
        
        while True:
            success, img = cap.read()
            if success:
                img = detector.findHolistics(img)
                
                cv.imshow("Image", img)
                k = cv.waitKey(1) & 0xFF
                if k == 27:
                    cap.release()
                    break
            else:
                print("No VideoCapture Detected :C")
                break;
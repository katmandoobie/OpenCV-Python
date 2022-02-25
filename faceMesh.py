import cv2 as cv
import mediapipe as mp

class faceMeshDetector():
    def __init__(self,mode = False, maxFaces = 2, trackingCon = 0.5,detectionCon = 0.5 ):
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        self.maxFaces = maxFaces
        self.mode = mode
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        #self.faceMesh = mpFaceMesh.FaceMesh()
        
    def findFaceMesh(self, img, draw = True):
        return img
        
    def displayFaceMesh():
        cap = cv.VideoCapture(0)
        detector = faceMeshDetector()
        while True:
            success, img = cap.read()
            if success:
                img = detector.findFaceMesh(img)
            
            
                cv.imshow("Image", img)
                k = cv.waitKey(1) & 0xFF
                if k == 27:
                    break
            else:
                print("No VideoCapture Detected :C")
                break;
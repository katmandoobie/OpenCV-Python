import cv2 as cv
import mediapipe as mp
import os

class faceMeshDetector():
    def __init__(self,mode = False, maxFaces = 2, trackingCon = 0.5,detectionCon = 0.5 ):
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        self.maxFaces = maxFaces
        self.mode = mode
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(maxFaces)
        self.drawSpec = self.mpDraw.DrawingSpec(color = (255, 0 ,0),
                                                thickness = 1, circle_radius = 1)
        
    def findFaceMesh(self, img, draw = True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    #self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION,
                    #                       self.drawSpec,self.drawSpec)
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                           self.drawSpec,self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    face.append([id,x,y])
                faces.append(face)
        return img, faces
        
    def displayFaceMesh():
        cap = cv.VideoCapture(0)
        detector = faceMeshDetector()
        faces = [] 
        while True:
            success, img = cap.read()
            if success:
                img, faces = detector.findFaceMesh(img)
                #print(faces) 
            
                cv.imshow("Image", img)
                k = cv.waitKey(1) & 0xFF
                if k == 27:
                    break
            else:
                print("No VideoCapture Detected :C")
                break;
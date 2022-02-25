import cv2 as cv
from matplotlib.transforms import Bbox
import mediapipe as mp

class faceDetector(): 
    def __init__(self, mode = False, maxFaces = 2, detectionCon = 0.5,
                 trackCon = 0.5):
        self.maxFaces = maxFaces
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.detectionCon)
    
    def findFaces(self, img, draw = True):
        imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                #print(detection.score)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin *ih), int(bboxC.width *iw),int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                #cv.rectangle(img, bbox, (255,0,255), 2)
                if draw:
                    img = self.fancyDraw(img,bbox)
                    cv.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1] - 20), cv.FONT_HERSHEY_PLAIN,
                           2, (255,0,255), 2)
        return img, bboxs
    
    def fancyDraw(self, img, bbox, l = 30, t = 5, rt = 1):
        x, y, w , h = bbox 
        x1, y1, = x+w, y+h
        
        cv.rectangle(img, bbox, (255,0,255), rt)
        #top left corner
        cv.line(img, (x,y),(x+l,y),(255,0,255),t)
        cv.line(img, (x,y),(x,y+l),(255,0,255),t)
        #top right corner
        cv.line(img, (x1,y),(x1-l,y),(255,0,255),t)
        cv.line(img, (x1,y),(x1,y+l),(255,0,255),t)
        #bottom left corner
        cv.line(img, (x,y1),(x+l,y1),(255,0,255),t)
        cv.line(img, (x,y1),(x,y1-l),(255,0,255),t)
        #bottom right corner
        cv.line(img, (x1,y1),(x1-l,y1),(255,0,255),t)
        cv.line(img, (x1,y1),(x1,y1-l),(255,0,255),t)
        return img
        
        
        
    def displayFaceDetection():
        cap = cv.VideoCapture(0)
        detector = faceDetector()
        while True:
            success, img = cap.read()
            if success:
                img , bboxs = detector.findFaces(img)
            
            
                cv.imshow("Image", img)
                k = cv.waitKey(1) & 0xFF
                if k == 27:
                    break
            else:
                print("No VideoCapture Detected :C")
                break;
            
            
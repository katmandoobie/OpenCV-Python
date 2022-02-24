from unittest import result
import cv2 as cv
import mediapipe as mp

class handDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5,
                 trackCon = 0.5): 
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands() #self.mode,self.maxHands, self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img,draw = True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:      
                if draw:    
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo = 0, draw =True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                    cv.circle(img, (cx,cy),10, (255,0,0),cv.FILLED)      
        return lmList
    
    def displayTracking():
        cap =  cv.VideoCapture(0)
        detector = handDetector()
        while True:
            success, img = cap.read()
            if success:
                #will find all hands in frame
                #draw = True by default
                img = detector.findHands(img)
                #will return positions of all landmarks
                #draw = True by default
                lmList  = detector.findPosition(img) 
                
                cv.imshow("Image", img)
                k = cv.waitKey(5) & 0xFF
                if k == 27:
                    break
            else:
                print("No VideoCapture Detected :C")
                break;
        
        
        

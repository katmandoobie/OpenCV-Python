from handtracking import handDetector
from poseEstimation import poseDetector
from faceDetection import faceDetector
from faceMesh import faceMeshDetector
from holistic import holisticDetector
import cv2 as cv
import sys


def main():
    args = sys.argv[1:]
    if args[0] == "ht":
        handDetector.displayTracking()
    elif args[0] == "pe":
        poseDetector.displayPoseEstimation()
    elif args[0] == "fd":
        faceDetector.displayFaceDetection()
    elif args[0] == "fm":
        faceMeshDetector.displayFaceMesh()
    elif args[0] == 'hol':
        holisticDetector.displayHolistics()
        
        
    cv.destroyAllWindows()
    

if __name__ == "__main__":
    main()
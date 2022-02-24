from handtracking import handDetector
from poseEstimation import poseDetector
import cv2 as cv
import sys


def main():
    args = sys.argv[1:]
    if args[0] == "ht":
        handDetector.displayTracking();
    elif args[0] == "pe":
        poseDetector.displayPoseEstimation(); 
        
    cv.destroyAllWindows()
    

if __name__ == "__main__":
    main()
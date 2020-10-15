import numpy as np
import argparse
import imutils
import dlib
import cv2
from scipy.spatial import distance
from imutils import face_utils

def sleepiness_detection():
  """Detects if a person feeling sleepy or not by evaluating EAR value"""
  face_detector = dlib.get_frontal_face_detector()
  # .dat file is pre-trained model for face detection 
  # available at http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 
  face_shape_predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
  (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
  (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

if __name__=='__main__':
  sleepiness_detection()
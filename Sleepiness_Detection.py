import numpy as np
import argparse
import imutils
import dlib
import cv2
from scipy.spatial import distance
from imutils import face_utils

def get_EAR(eye):
    """
    Based on the work by Soukupová and Čech in their 2016 paper,
    http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
    """
    return (distance.euclidean(eye[1], eye[5])+distance.euclidean(eye[2], eye[4]))/(2*distance.euclidean(eye[0], eye[3]))

def draw_contour(frame, landmark, clr=(0,255,0)):
    """ Draws contours on the given landmark, green is the default tone"""
    landmarkHull = cv2.convexHull(landmark)
    cv2.drawContours(frame, [landmarkHull], -1, clr, 1)

def sleepiness_detection():
    """Detects if a person feeling sleepy or not by evaluating EAR value"""
    face_detector = dlib.get_frontal_face_detector()
    # .dat file is pre-trained model for face detection
    # available at http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    face_shape_predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

    stream=cv2.VideoCapture(0)
    frame_counter=0

    while True:
        ret, frame = stream.read()
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        for face in faces:
            face_contour = face_shape_predictor(gray, face)
            face_contour = face_utils.shape_to_np(face_contour)

            leftEye, rightEye = face_contour[lStart:lEnd], face_contour[rStart:rEnd]
            leftEAR, rightEAR = get_EAR(leftEye), get_EAR(rightEye)
            EAR = (leftEAR + rightEAR) / 2.0

            draw_contour(frame, leftEye)
            draw_contour(frame, rightEye)

            if EAR < thresh:
                frame_counter += 1
                print(frame_counter)

        """To DO ALERT DISPLAY"""
            else:
                frame_counter = 0
        cv2.imshow("Main Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    stream.stop()


if __name__=='__main__':
    sleepiness_detection()

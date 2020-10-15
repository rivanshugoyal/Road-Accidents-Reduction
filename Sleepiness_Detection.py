import numpy as np
import argparse
import imutils
import dlib
import cv2
from scipy.spatial.distance import euclidean
from imutils import face_utils

def get_EAR(eye_idxs):
    val = (euclidean(eye_idxs[1], eye_idxs[5])+euclidean(eye_idxs[2], eye_idxs[4]))/2
    return val/euclidean(eye_idxs[0], eye_idxs[3])

def draw_contour(frame, landmark, clr=(0,255,0)):
    """ Draws contours on the given landmark, green is the default tone"""
    landmarkHull = cv2.convexHull(landmark)
    cv2.drawContours(frame, [landmarkHull], -1, clr, 1)

def sleepiness_detection(thresh, eye_check, face_check):
    """Detects if a person feeling sleepy or not by evaluating EAR_avg value"""

    face_detector = dlib.get_frontal_face_detector()
    face_shape_predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
    (l_begin, l_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (r_begin, r_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

    stream=cv2.VideoCapture(0)
    eye_frame_counter, face_frame_counter = 0, 0

    while True:
        ret, frame = stream.read()
        frame = imutils.resize(frame, width=500)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray_frame)

        if len(faces) == 0:
            face_frame_counter += 1
            print("#Frames when eyes not on the road: ", face_frame_counter)
            if face_frame_counter >= face_check:
                cv2.putText(frame, "Eyes not on the Road!", (10,325),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)
                print("\a Look front!")
        else:
            face_frame_counter = 0

        for face in faces:
            face_contour = face_shape_predictor(gray_frame, face)
            face_contour = face_utils.shape_to_np(face_contour)

            left_eye, right_eye = face_contour[l_begin:l_end], face_contour[r_begin:r_end]
            left_eye_EAR, right_eye_EAR = get_EAR(left_eye), get_EAR(right_eye)
            EAR_avg = (left_eye_EAR + right_eye_EAR) / 2

            draw_contour(frame, left_eye)
            draw_contour(frame, right_eye)

            if EAR_avg < thresh:
                eye_frame_counter += 1
                print("#Frames when eyes were not open: ", eye_frame_counter)
                if eye_frame_counter >= eye_check:
                    cv2.putText(frame, "********* EAR = "+str(round(EAR_avg, 3))+" *********", (10,325),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 1)
                    print("\a Wake Up!")
            else:
                eye_frame_counter = 0
        cv2.imshow("Main Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    stream.stop()

def main():
    parser = argparse.ArgumentParser(description="Laziness discovery utilizing pre-trained face indicator")
    parser.add_argument("--thresh",help="Threshold value of EAR_avg", type=int, default=0.25)
    parser.add_argument("--ffc",help="Threshold value of face frame counter", type=int, default=200)
    parser.add_argument("--efc",help="Threshold value of eye frame counter", type=int, default=45)
    args = parser.parse_args()

    sleepiness_detection(args.thresh, args.efc, args.ffc)

if __name__=='__main__':
    main()
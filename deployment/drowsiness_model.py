import dlib
import imutils

import numpy as np
import cv2 as cv

from imutils import face_utils
from scipy.spatial import distance as dist


class WrapperDrowsinessModel:
    def __init__(self):
        super(WrapperDrowsinessModel, self).__init__()
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predict = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.thresh = 0.35

    @staticmethod
    def _preprocess(frame: np.ndarray):
        frame = imutils.resize(frame, width=640)
        img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        return img_gray

    def predict(self, frame):
        preprocessed_frame = self._preprocess(frame)
        faces = self.face_detector(preprocessed_frame)
        if len(faces) == 0:
            return 1
        else:
            face = faces[0]

        landmarks = self.landmark_predict(preprocessed_frame, face)
        drowsy = self._postprocess(landmarks)

        # # -------------------
        # cv.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 5)
        # cv.rectangle(frame, left_p1, left_p2, (255, 0, 0), 5)
        # cv.rectangle(frame, right_p1, right_p2, (255, 0, 0), 5)
        # if drowsy:
        #     cv.putText(frame, 'Drowsy', (30, 30), cv.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
        # cv.imshow("Video", frame)

        return drowsy

    def _postprocess(self, landmarks):
        (L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
        shape = face_utils.shape_to_np(landmarks)

        # parsing the landmarks list to extract left eye and right eye landmarks
        left_eye = shape[L_start: L_end]
        right_eye = shape[R_start:R_end]

        # # ------------------------------
        # left_p1, left_p2 = self._get_bounding_box(left_eye)
        # right_p1, right_p2 = self._get_bounding_box(right_eye)

        left_EAR = self._calculate_ear(left_eye)
        right_EAR = self._calculate_ear(right_eye)
        avg = (left_EAR + right_EAR) / 2

        return avg < self.thresh

    @staticmethod
    def _calculate_ear(eye):
        # calculate the vertical distances
        y1 = dist.euclidean(eye[1], eye[5])
        y2 = dist.euclidean(eye[2], eye[4])

        # calculate the horizontal distance
        x1 = dist.euclidean(eye[0], eye[3])

        # calculate the EAR
        EAR = (y1 + y2) / x1
        return EAR

    @staticmethod
    def _get_bounding_box(eye):
        p1 = np.min(eye, axis=0)
        p2 = np.max(eye, axis=0)

        return p1 - 20, p2 + 20

import cv2 as cv

from time import time
from distraction_model import WrapperDistractionModel


def main():
    distraction_capture = cv.VideoCapture(2)
    distraction_model = WrapperDistractionModel()
    last_attention = time()
    while True:
        _, frame = distraction_capture.read()
        distraction_prediction = distraction_model.predict(frame)
        print(distraction_prediction)
        if distraction_prediction == 0:
            last_attention = time()

        else:
            non_attention_interval = time() - last_attention
            print(non_attention_interval)
            if non_attention_interval > 4:
                print("AUTO BREAK!!!!")

            elif non_attention_interval > 2:
                print("WARNING!!!")

        cv.imshow("stream", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()

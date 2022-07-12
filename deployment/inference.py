import sys
import can

import cv2 as cv

from time import time
from distraction_model import WrapperDistractionModel
from drowsiness_model import WrapperDrowsinessModel


def main():
    distraction_capture = cv.VideoCapture(int(sys.argv[1]))
    distraction_model = WrapperDistractionModel()

    drowsiness_capture = cv.VideoCapture(int(sys.argv[2]))
    drowsiness_model = WrapperDrowsinessModel()
    bus = can.Bus(channel='can0', interface='socketcan')
    stop_msg = can.Message(arbitration_id=0x054, data=[1], is_extended_id=False)
    cont_msg = can.Message(arbitration_id=0x054, data=[0], is_extended_id=False)

    last_attention = time()
    while True:
        _, distraction_frame = distraction_capture.read()
        distracted = distraction_model.predict(distraction_frame)

        _, drowsiness_frame = drowsiness_capture.read()
        drowsy = drowsiness_model.predict(drowsiness_frame)

        if not distracted and not drowsy:
            last_attention = time()
            bus.send(cont_msg)

        else:
            non_attention_interval = time() - last_attention
            print(non_attention_interval)
            if non_attention_interval > 4:
                print("AUTO BREAK!!!!")
                bus.send(stop_msg)

            elif non_attention_interval > 2:
                print("WARNING!!!")
                bus.send(cont_msg)

        # cv.imshow("stream", drowsiness_frame)
        # if cv.waitKey(1) & 0xFF == ord('q'):
        #     break


if __name__ == "__main__":
    main()

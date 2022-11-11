import cv2 as cv
import glob

import pandas as pd


def main():
    df = pd.read_csv("new_train.csv")

    for i, row in df.iloc[441:].iterrows():
        image_path = row['image_path']
        image = cv.imread(image_path)
        comp_image = cv.resize(image, (640, 480), interpolation=cv.INTER_AREA)
        cv.imshow("Image", comp_image)
        key = chr(cv.waitKey(0))
        df.loc[i, "label"] = int(key)
        comp_image_path = image_path.replace("New", "New Compressed")
        cv.imwrite(comp_image_path, comp_image)
        df.loc[i, "image_path"] = comp_image_path.replace("../", "")
        df.to_csv("new_train.csv", index=False)


if __name__ == "__main__":
    main()


import pandas as pd
import matplotlib.pyplot as plt
import skimage.io as io
import re


label_map = {
    'c0': 'safe driving',
    'c1': 'texting -right',
    'c2': 'talking on the phone -right',
    'c3': 'texting -left',
    'c4': 'talking on the phone -left',
    'c5': 'operating the radio',
    'c6': 'Drinking',
    'c7': 'reaching behined',
    'c8': 'hair and makeup',
    'c9': 'talking to passenger'
}
val_df = pd.read_csv('D:/4th Computer year/Graduation project/data/val.csv')
src_dir = 'D:/4th Computer year/Graduation project/'
dst_dir = 'D:/4th Computer year/Graduation project/bad_val'
for i in range(len(val_df)):
    print(i+1)
    img = io.imread('D:/4th Computer year/Graduation project/' + val_df['image_path'][i])
    class_= re.search('c\d',val_df['image_path'][i]).group()
    plt.axis('off')
    plt.title(f"label: {val_df['label'][i]}        class: {label_map[class_]}")
    ax = plt.gca()
    fig = plt.gcf()
    implot = ax.imshow(img)
    plt.show()





import cv2
import torchvision
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
from torch import nn
#from torch2trt import torch2trt
from distraction_model import DistractionModel
import time

def main():
    classes = [
    "Drive safe",
    "Text left",
    "Talk left",
    "Text right",
    "Talk right",
    "Adjust radio",
    "Drink",
    "Hair and makeup",
    "Reaching behind",
    "Talk to passanger"
    ]
    Alarm_classes = [
    "Text left",
    "Talk left",
    "Text right",
    "Talk right",
    "Adjust radio",
    "Drink",
    "Hair and makeup",
    "Reaching behind",
    "Talk to passanger"
    ]
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DistractionModel()
    model.load_state_dict(torch.load('model.pth', map_location=device))



    model.to(device)
    model.eval()

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    image_transforms = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])
    def classify(model,image_transforms,image_path,classes):
        model = model.eval()
        image = Image.open(image_path)
        image = image_transforms(image).float()
        image = image.unsqueeze(0)
        output = model(image)
        # print(output)
        _, predicted = torch.max(output.data,1)
        print(classes[predicted.item()])
        if classes[predicted.item()] in  Alarm_classes:
            print('Alerm')
            start = time.time()
            print(start)
            while(classes[predicted.item()] in  Alarm_classes):
                print("hello from while")
                stop = time.time()
                print("stop time = ", stop)
                print("The time of the run:", stop - start)
                if stop - start in  range(2.0,3.0):
                    print("Laod Noise",stop - start)
                    break
            print("The time of the run:", stop - start)
        else:
            print("safe")
    classify(model,image_transforms,"2.jpg",classes)
    
    cap = cv2.VideoCapture(0)
    counter = 0


    while(True):
    # Capture frame-by-frame
        ret, frame = cap.read()
    
    
        if ret:
                # display the frame in a window called frame
                cv2.imshow('frame', frame)
                k = cv2.waitKey(1)

                if k % 256 == 27:
                    print("closing")
                    break
                else:
                    img = "frame_{}.png".format(counter)
                    cv2.imwrite(img,frame)
                    classify(model, image_transforms,  img, classes)
                    counter = counter + 1
# When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

  #  print(device)
 #   shape = (1, 3, 480, 640)
 #   x = torch.ones(shape).to(device)

  #  trt_model = torch2trt(model, [x], fp16_mode=True)
   # torch.save(trt_model.state_dict(), 'trt_model.pt')


if __name__ == "__main__":
    main()

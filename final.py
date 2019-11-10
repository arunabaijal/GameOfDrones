from BingImages import BingImages
import cv2
import os
import requests
import shutil
from classifier import classify
import threading
import tarfile
import zipfile
from copy import deepcopy

class ObjectClassifier:

    # Constructor: What classes are to classify?
    #def __init__(self)

    # @PARAM count: number. The number of images to download for each class
    # @PARAM download: boolean. Download new training data to classify the model
    # @PARAM delete:  boolean. Delete existing training data and start from scratch
    def trainModel(self, count, download=True, delete=False):

        # Delete existing resources
        if(delete and download):
            try:
                shutil.rmtree("./tf")
                shutil.rmtree("./training_images")
                shutil.rmtree("./tmp")
            except FileNotFoundError:
                pass
        print("--Start training Model")

        folders = ["./tf", "./tf/training_data", "./tf/training_output", "./tf/training_data/summaries/basic"]
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # Retrain the tensorflow model
        print("--Training model")
        os.system("python3 retrain.py --tf/training_data/bottleneck_dir=bottlenecks --model_dir=tf/training_data/inception --summaries_dir=tf/training_data/summaries/basic --output_graph=tf/training_output/retrained_graph.pb --output_labels=tf/training_output/retrained_labels.txt --image_dir=training_images --how_many_training_steps=4000")

    def liveDetect(self):
        pic_num = 1
        for imagename in os.listdir("/home/aruna/testImages"):
            try:
                #urllib.request.urlretrieve(i, "neg/"+str(pic_num)+".jpg")
                path = r'/home/aruna/testImages/' + imagename
                img1 = cv2.imread(path,cv2.IMREAD_COLOR)
                img = img1[60:228, 60:372]              
                cv2.imshow("image", img)
                # Break if input key equals "ESC"
                cv2.waitKey()
                # should be larger than samples / pos pic (so we can place our image on it)
                #resized_image = cv2.resize(img, (720, 720))
                #cv2.imwrite("outputImages/"+str(pic_num)+".jpg",cropped_image)
                pic_num += 1
                filename = "/tmp/object.jpg"

                # Inner function for thread to parallel process image classification according to trained model
                def classifyFace():
                    print("Classifying feature")
                    os.system("python -W ignore /home/aruna/Downloads/Final/label_image.py --graph=./output/retrained_graph.pb --labels=./output/retrained_labels.txt --input_layer=Placeholder --output_layer=final_result --image=" + filename)

                    # prediction = classify(filename, "./output/retrained_graph.pb", "./output/retrained_labels.txt", shape=720)
                    # nonlocal text
                    # text = prediction[0][0]
                    print("Finished classifying with text: " + text)

                # Standard text that is displayed above recognized face
                text = "unidentified object"
                exceptional_frames = 100
                startpoint = (0, 0)
                endpoint = (0, 0)
                color = (0, 0, 255) # Red
                # Read frame from camera stream and convert it to greyscale
                #img = cv2.imread(input_image)
                orig_image=img.copy()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)

                contours, hierarchy=cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
                cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse = True)

                # Loop through detected faces and set new face rectangle positions
                for c in cntsSorted[1:3]:
                    x,y,w,h=cv2.boundingRect(c)
                    color = (0, 0, 255)
                    if not text == "unidentified object":
                        color = (0, 255, 0)
                    oldstartpoint = deepcopy(startpoint)
                    startpoint = (x, y)
                    endpoint = (x + w, y + h)
                    object1 = (img[y:y + h, x:x + w])
                    cv2.imwrite(filename, object1)
                    threading._start_new_thread(classifyFace, ())
                    # Draw face rectangle and text on image frame
                    cv2.rectangle(img, startpoint, endpoint, color, 2)
                    textpos = (startpoint[0], startpoint[1] - 7)
                    cv2.putText(img, text, textpos, 1, 1.5, color, 2)

                # Show image in cv2 window
                cv2.imshow("image", img)
                # Break if input key equals "ESC"
                cv2.waitKey()
                # exceptional_frames += 1

            except Exception as e:
                print(str(e))

if __name__ == "__main__":
    classifier = ObjectClassifier()
    # classifier.trainModel(100, download=True, delete=True)
    classifier.liveDetect()
    # classifier.fastenNetwork(newDataset=False)
    # classifier.trainFast()
    # classifier.liveDetect()
    # classifier.liveFaceDetection()


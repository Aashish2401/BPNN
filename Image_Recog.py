import cv2
import numpy as np
from PIL import Image
from keras.models import load_model

digits = ['0','1','2','3','4','5','6','7','8','9']
letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

im = cv2.imread('D:/Work/Python/DMS/Assignment_2/Group.png') #Reads the image with written text containing 5 characters

h,w,c = im.shape    #Store size of image

img = np.zeros((h,w))

#Converting to numpy array of float values for further operations 
for i in range(h):
    for j in range(w):
        img[i][j] = float((255 - im[i][j][0]) / 255)
        
start = []
end = []
flag = 0

#A primitive way of splitting the image into 5 separate characters to recognise separately
for i in range(w):
    sums = 0
    for j in range(h):
        sums = sums + img[j][i]
    if sums != 0:
        if flag == 0:
            start.append(i-2)
            flag = 1
    if sums == 0:
        if flag == 1:
            end.append(i+1)
            flag = 0
            
new_img = [[[(255,255,255)]*28]*h]*len(start)
new_img = np.asarray(new_img)
for ind in range(len(start)):
    for i in range(h):
        for j in range(start[ind],end[ind]):
            new_img[ind][i][j - start[ind] + 5] = im[i][j]

#Saving the split up images locally
for i in range(len(start)):
    cv2.imwrite("D:/Work/Python/DMS/Assignment_2/{0}.png".format(i),new_img[i])

answer = []

model1 = load_model("D:/Work/Python/DMS/Assignment_2/Letter_model.h5")  #importing the model to recognize letters
model2 = load_model("D:/Work/Python/DMS/Assignment_2/Digit_model.h5")   #importing the model to recognize digits

img=0
#Recognizing the letters first
for i in range(3):
        x=np.zeros((1,784))
        img= Image.open("D:/Work/Python/DMS/Assignment_2/{0}.png".format(i))
        arr=list(img.getdata())
        for i in range(784):
            x[0][i] = (255-arr[i][0])/255
        result = model1.predict(x)
        answer.append(letters[np.argmax(result)-1])

#Recognizing the digits
for i in range(3,5):
        x=np.zeros((1,784))
        img= Image.open("D:/Work/Python/DMS/Assignment_2/{0}.png".format(i))
        arr=list(img.getdata())
        for i in range(784):
            x[0][i] = (255-arr[i][0])/255
        result = model2.predict(x)
        answer.append(digits[np.argmax(result)])

print(answer)   #Printing the result

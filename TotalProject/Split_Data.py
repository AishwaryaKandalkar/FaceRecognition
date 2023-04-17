import numpy as np
import os
import tqdm
import cv2

data=[]
for img in os.listdir("data"):
        #returns a list of images in that directery
        path=os.path.join("data",img)
        # os. path. join combines path names into one complete path. 
        # This means that you can merge multiple parts of a path into one, 
        # instead of hard-coding every path name manually.
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # imread() method loads an image from the specified file.
        img_data = cv2.resize(img_data, (50,50))
        name = img.split('.')[-3] 
        data.append([np.array(img_data), np.array(int(name))])
        
# print(data)

train = data[:8]  
# everything before, but not including, the index.
test = data[8:]
# everything after the specified index.
X_train = np.array([i[0] for i in train]).reshape(-1,50,50,1)
# -1: automatically calculates no of image
# image size 50,50
# 1: grayscale image
print(X_train.shape)
y_train = [i[1] for i in train]
X_test = np.array([i[0] for i in test]).reshape(-1,50,50,1)
print(X_test.shape)
y_test = [i[1] for i in test]
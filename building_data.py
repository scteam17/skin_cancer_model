"""******************************************************************************************************************
**this program take the images for the data folder and convert them to numpy arrays of size(32,32,3) and save them  *
**in the training folder                                                                                            *         
**                                                                                                                  *
**pls make sure to create a folder called training to save the numpy arrays in it also make sure that               * 
**    the data folder is the same directory as this file                                                            *
**                                                                                                                  *
******************************************************************************************************************"""

import os
import numpy as np
from PIL import Image




#image path
folder_benign_train = 'data/train/benign'
folder_malignant_train = 'data/train/malignant'

folder_benign_test = 'data/test/benign'
folder_malignant_test = 'data/test/malignant'



#to read the image then resize them
read = lambda imname: np.asarray(Image.open(imname).convert("RGB").resize((64,64)))




#loading the training data
print("loading the training data....")
ims_benign = [read(os.path.join(folder_benign_train, filename)) for filename in os.listdir(folder_benign_train)]
X_benign_train = np.array(ims_benign, dtype='uint8')
ims_malignant = [read(os.path.join(folder_malignant_train, filename)) for filename in os.listdir(folder_malignant_train)]
X_malignant_train = np.array(ims_malignant, dtype='uint8')
print("training data loaded successfully")


# Load in testing data
#note : using the same ims_bengign and ims_malignant for saving the memory
print("loading the test data....")
ims_benign = [read(os.path.join(folder_benign_test, filename)) for filename in os.listdir(folder_benign_test)]
X_benign_test = np.array(ims_benign)
#X_benign_test = np.array(ims_benign, dtype='uint8')
ims_malignant = [read(os.path.join(folder_malignant_test, filename)) for filename in os.listdir(folder_malignant_test)]
X_malignant_test = np.array(ims_malignant)
#X_malignant_test = np.array(ims_malignant, dtype='uint8')
print("the test data loaded successfully")


#creating labels is simple 0 for the benign and 1 for malignant
print("creating training and test labels")
y_benign_train = np.zeros(X_benign_train.shape[0])
y_malignant_train = np.ones(X_malignant_train.shape[0])

y_benign_test = np.zeros(X_benign_test.shape[0])
y_malignant_test = np.ones(X_malignant_test.shape[0])


#putting the data togethor for the x_train and y_train then shuffle them
# Merge data
print("merging the data")
X_train = np.concatenate((X_benign_train, X_malignant_train), axis = 0)
y_train = np.concatenate((y_benign_train, y_malignant_train), axis = 0)

X_test = np.concatenate((X_benign_test, X_malignant_test), axis = 0)
y_test = np.concatenate((y_benign_test, y_malignant_test), axis = 0)


# Shuffle data
print("shuffling the data")
s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
y_train = y_train[s]

s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
y_test = y_test[s]


'''
#displaying the first 15 images
print("displaying the first 15 images")
w=40
h=30
fig=plt.figure(figsize=(12, 8))
columns = 5
rows = 3

for i in range(1, columns*rows +1):
    ax = fig.add_subplot(rows, columns, i)
    if y_train[i] == 0:
        ax.title.set_text('Benign')
    else:
        ax.title.set_text('Malignant')
    plt.imshow(X_train[i], interpolation='nearest')
plt.show()

'''


print("saving the data as numpy arrays")
np.save('./training/X_train',X_train)
np.save('./training/X_test',X_test)
print("traning and test data saved ")
np.save('./training/y_train',y_train)
np.save('./training/y_test',y_test)
print("training and test label saved")
print("all data saved")
print("program finishes")

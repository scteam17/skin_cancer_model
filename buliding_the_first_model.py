f'''****************************************************************
this model take the data from the training folder ,build the model*
then save it in the training folder again also plots the accuracy *
loss , recall and percision vs the epoch                          *
                                                                  *
NOTE ->   1)all of them is not that important only important part *
          is the recall plot this is the one we need to optimize  *
          2)once the plot is shown don't forget to save them      *
          remember you can't get them back once the program close *
          
input filters = 32
first model  => [batch size = 64]
second model => [batch size = 32]
third model  => [batch size = 32 , input filters = 64]
model no.4   => [add conv , i/p filters = 124]
model no.5   => [batch size  = 64]
model no.6   => [batch size  = 124] remove one conv 3conv layers
model no.7   => [batch size  = 124] 4conv layers
model no.8   => [batch size  = 124] 5conv layers
model no.9   => [batch size  = 124] 5conv layers with 224 filters
---------------------testing different optimizer(from adam to sgd)------------------
model no.10  => [batch size  = 124] 5conv layers with 224 filters 
                lr=0.01, momentum=0.9, decay=0.01
model no.11  => [batch size  = 124] 5conv layers with 224 filters
                lr=0.005, momentum=0.9, decay=0.01
model no.12  => [batch size  = 124, epoch = 150] 5conv layers with 224 filters 
                lr=0.005, momentum=0.9, decay=0.01
model no.13  => [batch size  = 124, epoch = 100] 5conv layers with 224 filters 
                lr=0.05, momentum=0.9, decay=0.01
model no.14  => [batch size  = 124, epoch = 200] 5conv layers with 224 filters)
                lr=0.05, momentum=0.9, decay=0.01
model no.15  => [batch size  = 124, epoch = 250] 5conv layers with 224 filters)
                lr=0.05, momentum=0.9, decay=0.01
model no.16  => [batch size  = 124, epoch = 100] 5conv layers with 224 filters)
                lr=0.01, momentum=0.5, decay=0.01
model no.17  => [batch size  = 124, epoch = 100] 5conv layers with 224 filters)
                lr=0.01, momentum=0.5, decay=0.05
model no.18  => [batch size  = 124, epoch = 200] 5conv layers with 224 filters)
                lr=0.01, momentum=0.5, decay=0.05
model no.19  => [batch size  = 124, epoch = 100] 5conv layers with 224 filters)
                lr=0.005, momentum=0.5, decay=0.05
                
                
                
****************************************************************'''


#------------------IMPORTS--------------------------------------------------------
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding

#------------------LOADING THE DATA------------------------------------------------

#loading
X_train = np.load('./training/X_train.npy')
X_test = np.load('./training/X_test.npy')
y_train = np.load('./training/y_train.npy')
y_test = np.load('./training/y_test.npy')

#encoding
y_train = to_categorical(y_train, num_classes= 2)
y_test = to_categorical(y_test, num_classes= 2)

# normalizing
X_train = X_train/255.
X_test = X_test/255

#------------------BUILDING THE MODEL----------------------------------------------

model = Sequential([
        Conv2D(248,kernel_size=(3, 3),padding='Same', activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(124,kernel_size=(3, 3),padding = 'Same',input_shape=(64,64,3),activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64,kernel_size=(3, 3),padding='Same', activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(32,kernel_size=(3, 3),padding='Same', activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(32,kernel_size=(3, 3),padding='Same', activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')
    ])


#------------------COMPILATION, FITTING AND SAVING THE MODEL-----------------------
'''****************************************************************
fitting the model using gpu if you want to use cpu just delete the*
with tf.device("/GPU:0"):                                         *  
****************************************************************'''

#try sgd optimzer
from keras.optimizers import SGD
opt = SGD(lr=0.005, momentum=0.9, decay=0.05)



#compiling
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), "acc"])
epochs = 100
batch_size = 124

#fitting
with tf.device("/GPU:0"):
    try:
        history = model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(X_test,y_test))
    except Exception as e:
        print(e)

#saving
model.save('./training/model19.h5')


#------------------PLOTTING-----------------------------------------------------------

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for recall
plt.plot(history.history['recall'])
plt.plot(history.history['val_recall'])
plt.title('model recall')
plt.ylabel('recall')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for precision
plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.title('model precision')
plt.ylabel('precision')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
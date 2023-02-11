import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.applications import ResNet50


#loading the data
print("loading the data............")
X_train = np.load('./training/X_train.npy')
X_test = np.load('./training/X_test.npy')
y_train = np.load('./training/y_train.npy')
y_test = np.load('./training/y_test.npy')
print("the data has been loaded successfully")


print("encoding the labels for predictions....")
y_train = to_categorical(y_train, num_classes= 2)
y_test = to_categorical(y_test, num_classes= 2)


# normalization for applying the data augmentation to prevent overfitting
print("normalizing the data.....")
X_train = X_train/255.
X_test = X_test/255

# See learning curve and validation curve

#building the first model
print("building the model......")


model = ResNet50(include_top=True,weights=None,input_tensor=None,input_shape=(32,32,3),pooling='avg',classes=2)

print("compile .......")
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), "acc"])
epochs = 100  #100
batch_size = 32  #32

print("fitting.........")

with tf.device("/GPU:0"):
    try:
        history = model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(X_test,y_test))
    except Exception as e:
        print(e)
print("saving the model...")
model.save('./training/theFirstModelnew4.h5')






# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('resnet model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('resnet model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for recall
plt.plot(history.history['recall'])
plt.plot(history.history['val_recall'])
plt.title('resnet model recall')
plt.ylabel('recall')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for precision
plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.title('resnet model precision')
plt.ylabel('precision')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
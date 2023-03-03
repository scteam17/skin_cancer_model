#------------------IMPORTS--------------------------------------------------------
from model import Model
import matplotlib.pyplot as plt
import numpy as np
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

#------------------paramaters----------------------------------------------------
opt = "adam"
epochs = 100
batch_size = 32
#dir = './training/model26.h5'
dir = './training/modeltest3.h5'
validation = (X_test,y_test)

#-------------------------running------------------------------------------------------
model = Model()
model.compile(opt)
model.fit(X_train,y_train,validation,batch_size,epochs)
model.save(dir)
history = model.history
print(history.history.keys())

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

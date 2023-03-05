#------------------IMPORTS--------------------------------------------------------
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.applications import ResNet152
from keras.applications.vgg19 import VGG19
from keras.applications.mobilenet import MobileNet
from keras.applications.xception import Xception
#------------------BUILDING THE MODEL----------------------------------------------
class Model:
    def __init__(self):
        self.model = self.mobileNet()
        self.history = ''

    def normal_model(self):
        return Sequential([
            self.data_augmentation(),
            #Conv2D(124,kernel_size=(3, 3),padding = 'Same', input_shape=(64,64,3),activation="relu"), #comment if you use augmnetation
            Conv2D(124,kernel_size=(3, 3),padding = 'Same',activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(64,kernel_size=(3, 3),padding = 'Same',activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(32,kernel_size=(3, 3),padding = 'Same',activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(16,kernel_size=(3, 3),padding='Same', activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Conv2D(16,kernel_size=(3, 3),padding='Same', activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(2, activation='softmax')
        ])

    def ResNet(self):  #TODO: try play with different parameters
        return ResNet152(include_top=True,weights=None,input_tensor=None,input_shape=(64,64,3),pooling='max',classes=2)

    def VGG(self):#TODO : try play with different parameters
        return VGG19(weights=None, input_shape=(64,64,3),classes=2)

    def mobileNet(self):#TODO: try play with different parameters
        return MobileNet(input_shape=(64,64,3),weights=None,classes=2)

    def xceptionNet(self):#TODO: try play with different parameters
        return Xception(weights=None,input_shape=(71,71,3),classes=2)

    def alexNet(self):#TODO: try play with different parameters
        pass

    def data_augmentation(self):
        return  Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", input_shape=(64,64,3)),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomRotation(0.2)
        ])

    def compile(self,opt):
        '''****************************************************************
        fitting the model using gpu if you want to use cpu just delete the*
        with tf.device("/GPU:0"):                                         *
        ****************************************************************'''
        self.model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=[
                               tf.keras.metrics.Recall()
                               ])

    def fit(self,X_train,y_train,validation,batch_size,epochs):
        with tf.device("/GPU:0"):
            try:
                self.history = self.model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=validation)
            except Exception as e:
                print(e)

    def save(self,dir):
        self.model.save(dir)
        print('model saved successfully ......')
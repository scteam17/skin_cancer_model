'''from keras.models import load_model
#loading the model
model = load_model('./training/theFirstModel.h5')

print(model.history)'''


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#print(tf.device("/GPU:0"))
tf.debugging.set_log_device_placement(True)

# Create some tensors
with tf.device("/GPU:0"):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)

print(c)
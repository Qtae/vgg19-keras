import tensorflow as tf
import tensorflow_addons as tfa

def conv_block(input, filters, conv_size=(3,3), padding='same', regularizer= tf.keras.regularizers.L2(0.01)):
  x = tf.keras.layers.Conv2D(filters, conv_size, padding=padding)(input)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tfa.activations.mish(x)
  return x

def vgg19(input):
  x = conv_block(input,64,(3,3))
  x = conv_block(x,64,(3,3))
  x = tf.keras.layers.MaxPool2D(padding='same')(x)

  x = conv_block(x,128,(3,3))
  x = conv_block(x,128,(3,3))
  x = tf.keras.layers.MaxPool2D(padding='same')(x)

  x = conv_block(x,256,(3,3))
  x = conv_block(x,256,(3,3))
  x = conv_block(x,256,(3,3))
  x = conv_block(x,256,(3,3))
  x = tf.keras.layers.MaxPool2D(padding='same')(x)

  x = conv_block(x,512,(3,3))
  x = conv_block(x,512,(3,3))
  x = conv_block(x,512,(3,3))
  x = conv_block(x,512,(3,3))
  x = tf.keras.layers.MaxPool2D(padding='same')(x)

  x = conv_block(x,512,(3,3))
  x = conv_block(x,512,(3,3))
  x = conv_block(x,512,(3,3))
  x = conv_block(x,512,(3,3))
  x = tf.keras.layers.MaxPool2D(padding='same')(x)

  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(4096, activation='relu')(x)
  x = tf.keras.layers.Dense(512, activation='relu')(x)
  output = tf.keras.layers.Dense(2, activation='softmax')(x)

  model = tf.keras.models.Model(inputs=input, outputs=output)

  return model

def inceptionV4(input):
  return
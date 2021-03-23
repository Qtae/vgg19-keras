import os
import cv2
import random
import numpy as np
import tqdm
import tensorflow as tf
import tensorflow_addons as tfa


def shuffle_2Dlist(_input):
  if (_input is None) or (len(_input[0]) == 0):
    return [[],[]]
  c= list(zip(_input[0],_input[1]))
  random.shuffle(c)
  a,b = zip(*c)
  return [list(a),list(b)]

def get_class_dirs(rootdir):
  return [os.path.join(rootdir, f) for f in os.listdir(rootdir) if os.path.isdir(os.path.join(rootdir, f))]

def get_dataset(cls_dirs, positive_label_name_list):
  trainset = [[],[]] # [[filepaths],[labels]]
  for class_dir in cls_dirs:
    if class_dir.split('\\')[-1] in positive_label_name_list:
      label = 0
    else:
      label = 1
    train_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith(('.png','jpeg','jpg','.bmp')) and ('Psr' in f) and (os.path.exists(os.path.join(class_dir, f.replace('Psr', 'Pad')))) and (os.path.exists(os.path.join(class_dir, f.replace('Psr', 'Axs'))))]
    train_labels = [label for f in train_files]
    trainset[0].extend(train_files)
    trainset[1].extend(train_labels)
  trainset = shuffle_2Dlist(trainset)
  return trainset

def load_data(dataset, normalize=True, aug_rot=False):
  if aug_rot:
    train_image_list=[]
    train_label_list=[]
    for f, lbl in zip(dataset[0], dataset[1]):
      if lbl == 0:
        img_psr = cv2.resize(cv2.imread(f, cv2.IMREAD_COLOR), dsize=(224, 224))
        img_pad = cv2.resize(cv2.imread(f.replace("Psr", "Pad"), cv2.IMREAD_GRAYSCALE), dsize=(224, 224))
        img_axs = cv2.resize(cv2.imread(f.replace("Psr", "Axs"), cv2.IMREAD_GRAYSCALE), dsize=(224, 224))
        train_image_list.append(np.concatenate((img_psr, np.expand_dims(img_pad, axis=2), np.expand_dims(img_axs, axis=2)), axis=2))
        train_label_list.append(lbl)
        img_psr = cv2.rotate(img_psr, cv2.ROTATE_90_CLOCKWISE)
        img_pad = cv2.rotate(img_pad, cv2.ROTATE_90_CLOCKWISE)
        img_axs = cv2.rotate(img_axs, cv2.ROTATE_90_CLOCKWISE)
        train_image_list.append(np.concatenate((img_psr, np.expand_dims(img_pad, axis=2), np.expand_dims(img_axs, axis=2)), axis=2))
        train_label_list.append(lbl)
        img_psr = cv2.rotate(img_psr, cv2.ROTATE_90_CLOCKWISE)
        img_pad = cv2.rotate(img_pad, cv2.ROTATE_90_CLOCKWISE)
        img_axs = cv2.rotate(img_axs, cv2.ROTATE_90_CLOCKWISE)
        train_image_list.append(np.concatenate((img_psr, np.expand_dims(img_pad, axis=2), np.expand_dims(img_axs, axis=2)), axis=2))
        train_label_list.append(lbl)
        img_psr = cv2.rotate(img_psr, cv2.ROTATE_90_CLOCKWISE)
        img_pad = cv2.rotate(img_pad, cv2.ROTATE_90_CLOCKWISE)
        img_axs = cv2.rotate(img_axs, cv2.ROTATE_90_CLOCKWISE)
        train_image_list.append(np.concatenate((img_psr, np.expand_dims(img_pad, axis=2), np.expand_dims(img_axs, axis=2)), axis=2))
        train_label_list.append(lbl)
      else:
        img_psr = cv2.resize(cv2.imread(f, cv2.IMREAD_COLOR), dsize=(224, 224))
        img_pad = cv2.resize(cv2.imread(f.replace("Psr", "Pad"), cv2.IMREAD_GRAYSCALE), dsize=(224, 224))
        img_axs = cv2.resize(cv2.imread(f.replace("Psr", "Axs"), cv2.IMREAD_GRAYSCALE), dsize=(224, 224))
        train_image_list.append(np.concatenate((img_psr, np.expand_dims(img_pad, axis=2), np.expand_dims(img_axs, axis=2)), axis=2))
        train_label_list.append(lbl)
        
    randomlist = shuffle_2Dlist([train_image_list, train_label_list])
    train_images = np.array(randomlist[0], dtype=np.uint8)
    train_labels = np.array(randomlist[1], dtype=np.uint8)
  else:
    train_images = np.array([np.concatenate((cv2.resize(cv2.imread(f, cv2.IMREAD_COLOR), dsize=(224, 224)), np.expand_dims(cv2.resize(cv2.imread(f.replace("Psr", "Pad"), cv2.IMREAD_GRAYSCALE), dsize=(224, 224)), axis=2), np.expand_dims(cv2.resize(cv2.imread(f.replace("Psr", "Axs"), cv2.IMREAD_GRAYSCALE), dsize=(224, 224)), axis=2)), axis=2) for f in dataset[0]], dtype=np.uint8)
    train_labels = np.array(dataset[1], dtype=np.uint8)
  if normalize:
    train_images = train_images.astype(np.float32)/ 255.
  return train_images, train_labels

def conv_block(input, filters, conv_size=(3,3), padding='same', regularizer= tf.keras.regularizers.L2(0.01)):
    x = tf.keras.layers.Conv2D(filters, conv_size, padding=padding)(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tfa.activations.mish(x)
    return x

with tf.device('/GPU:0'):
  ##VGG19##
  input = tf.keras.layers.Input((224,224,5))
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

  ##InceptionV4##
  #input = tf.keras.layers.Input((224,224,5))
  #output = 

  model = tf.keras.models.Model(inputs=input, outputs=output)
  adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
  #sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum = 0.2)
  model.compile(optimizer=adam,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy'])
  model.summary()

  rootdir = 'D:/Public/qtkim/PGM/20210420_KOR_Classification_DATA/Train'
  savedir = 'D:/Public/qtkim/PGM/Checkpoints/etch1'

  class_dirs = get_class_dirs(rootdir)

  positive_label_name_list = ['Etching']

  trainset = get_dataset(class_dirs, positive_label_name_list)
  train_images, train_labels = load_data(trainset)

  print(train_images.shape)
  print(train_labels.shape)

  filename = 'ADAM_{epoch:02d}-acc{val_accuracy:.2f}.hdf5'
  checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(savedir, filename),
                                                  verbose=1,
                                                  monitor='val_accuracy',
                                                  save_best_only=True,
                                                  mode='max')
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, mode='auto')
  reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2, cooldown=5, min_lr=0, mode='auto')#, verbose=1)
  callbacks_list = [checkpoint, early_stopping, reduce]
  model.fit(x=train_images, y= train_labels, validation_split=0.2, batch_size=16, epochs=500, callbacks=callbacks_list)
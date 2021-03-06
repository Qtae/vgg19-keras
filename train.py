import os
import argparse

import tensorflow as tf

from model import *
from data_manager import *


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('train_dir', type=str, help='train dataset directory')
  parser.add_argument('ckpt_dir', type=str, help='checkpoint save directory')
  parser.add_argument('positive_names', type=str, nargs='+', help='checkpoint save directory')
  args = parser.parse_args()

  with tf.device('/GPU:0'):
    ##VGG19##
    input = tf.keras.layers.Input((224,224,5))
    model = vgg19(input)
    adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
    #sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum = 0.2)
    model.compile(optimizer=adam,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    model.summary()

    rootdir = args.train_dir
    savedir = args.ckpt_dir

    class_dirs = get_class_dirs(rootdir)

    positive_label_name_list = args.positive_names

    trainset = get_dataset_5ch(class_dirs, positive_label_name_list)
    train_images, train_labels = load_data_5ch(trainset)

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
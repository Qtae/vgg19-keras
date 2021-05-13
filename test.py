import os
import cv2
import numpy as np
import tqdm
import tensorflow as tf

from data_manager import *

def test(model_path, rootdir, positive_label_name_list):
  with tf.device('/GPU:1'):
    model_name = model_path.split('/')[-2] + '_' + os.path.basename(model_path).split('.')[0] + os.path.basename(model_path).split('.')[1]
    model = tf.keras.models.load_model(model_path)
    #model.summary()

    resultdir = os.path.join('D:/Public/qtkim/PGM/TestResult','test')
    if not (os.path.exists(resultdir) and os.path.isdir(resultdir)):
      os.makedirs(resultdir)

    class_dirs = get_class_dirs(rootdir)

    testset = get_dataset(class_dirs, positive_label_name_list)
    test_images, test_labels = load_data(testset, True, False)
    
    softmax_arr = model.predict(test_images)

    print('Current Model : ',model_name)
    print('Total Test Images : ',softmax_arr.shape[0])

    for i, softmax in enumerate(tqdm.tqdm(softmax_arr, ncols=80)):
      img = test_images[i] * 255.
      true = test_labels[i]
      pred = np.argmax(softmax)
      image_psr = img[:, :, 0:3]
      image_pad = img[:, :, 3]
      image_axs = img[:, :, 4]
      resultdir_0 = os.path.join(resultdir, '0')
      resultdir_1 = os.path.join(resultdir, '1')
      if not (os.path.exists(resultdir_0) and os.path.isdir(resultdir_0)):
        os.makedirs(resultdir_0)
      if not (os.path.exists(resultdir_1) and os.path.isdir(resultdir_1)):
        os.makedirs(resultdir_1)
      
      if pred == 0:
        cv2.imwrite(os.path.join(resultdir_0, ('[Mismatch]' + '[' + str(softmax[0]) + ',' + str(softmax[1]) + ']' if (true == 1) else '') + str(i) + '_psr.jpeg'), image_psr)
        cv2.imwrite(os.path.join(resultdir_0, ('[Mismatch]' + '[' + str(softmax[0]) + ',' + str(softmax[1]) + ']' if (true == 1) else '') + str(i) + '_pad.jpeg'), image_pad)
        cv2.imwrite(os.path.join(resultdir_0, ('[Mismatch]' + '[' + str(softmax[0]) + ',' + str(softmax[1]) + ']' if (true == 1) else '') + str(i) + '_axs.jpeg'), image_axs)
      elif pred == 1:
        cv2.imwrite(os.path.join(resultdir_1, ('[Mismatch]' + '[' + str(softmax[0]) + ',' + str(softmax[1]) + ']' if (true == 0) else '') + str(i) + '_psr.jpeg'), image_psr)
        cv2.imwrite(os.path.join(resultdir_1, ('[Mismatch]' + '[' + str(softmax[0]) + ',' + str(softmax[1]) + ']' if (true == 0) else '') + str(i) + '_pad.jpeg'), image_pad)
        cv2.imwrite(os.path.join(resultdir_1, ('[Mismatch]' + '[' + str(softmax[0]) + ',' + str(softmax[1]) + ']' if (true == 0) else '') + str(i) + '_axs.jpeg'), image_axs)


if __name__=='__main__':
  test('D:/Public/qtkim/PGM/Checkpoints/etch1/ADAM_09-acc0.99.hdf5', 'D:/Public/qtkim/PGM/20210420_KOR_Classification_DATA/Test', ['Etching'])
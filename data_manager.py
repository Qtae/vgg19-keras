import os
import random
import numpy as np
import cv2

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
    train_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith(('.png','jpeg','jpg','.bmp'))]
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
        img = cv2.resize(cv2.imread(f, cv2.IMREAD_COLOR), dsize=(224, 224))
        train_image_list.append(img)
        train_label_list.append(lbl)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        train_image_list.append(img)
        train_label_list.append(lbl)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        train_image_list.append(img)
        train_label_list.append(lbl)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        train_image_list.append(img)
        train_label_list.append(lbl)
      else:
        img = cv2.resize(cv2.imread(f, cv2.IMREAD_COLOR), dsize=(224, 224))
        train_image_list.append(img)
        train_label_list.append(lbl)
        
    randomlist = shuffle_2Dlist([train_image_list, train_label_list])
    train_images = np.array(randomlist[0], dtype=np.uint8)
    train_labels = np.array(randomlist[1], dtype=np.uint8)
  else:
    train_images = np.array([cv2.resize(cv2.imread(f, cv2.IMREAD_COLOR), dsize=(224, 224)) for f in dataset[0]], dtype=np.uint8)
    train_labels = np.array(dataset[1], dtype=np.uint8)
  if normalize:
    train_images = train_images.astype(np.float32)/ 255.
  return train_images, train_labels

def get_dataset_5ch(cls_dirs, positive_label_name_list):
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

def load_data_5ch(dataset, normalize=True, aug_rot=False):
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

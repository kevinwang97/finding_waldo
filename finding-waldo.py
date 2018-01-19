import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import sys
import re
import json
import random

WALDO_LOCS_FILE = 'waldo_locations.json'
TRAINING_PATH = 'images/training/'
GENERATED_PIC_HEIGHT = 224
GENERATED_PIC_WIDTH = 224

def is_overlapping(p1, p2):
  # indices
  # bottom left x: 0
  # bottom left y: 1
  # top right x: 2
  # top right y: 3
  return not (p1[2] < p2[0] or p1[0] > p2[2] or p1[3] < p2[1] or p1[1] > p2[3])

def generate_train_data(num_with_waldo, num_without_waldo_per_waldo):
  images_np = []
  mask_images_np = []
  with open(WALDO_LOCS_FILE, 'r') as f:
    waldo_locations = json.load(f)
  keys = random.sample(waldo_locations, num_with_waldo)
  for name in keys:
    print "sampling from {}".format(name)
    locs = waldo_locations[name]
    rand_loc = random.choice(locs)[:]
    #image with waldo
    im = Image.open(TRAINING_PATH + name + '.jpg')
    height, width = im.size
    crop_height = rand_loc[3] - rand_loc[1]
    crop_width = rand_loc[2] - rand_loc[0]
    # if the images are smaller than the 
    if crop_height < GENERATED_PIC_HEIGHT:
      amt_add_top = random.randint(0, GENERATED_PIC_HEIGHT - crop_height)
      amt_add_bottom = GENERATED_PIC_HEIGHT - crop_height - amt_add_top
      rand_loc[1] -= amt_add_bottom
      rand_loc[3] += amt_add_top
    if crop_width < GENERATED_PIC_WIDTH:
      amt_add_left = random.randint(0, GENERATED_PIC_WIDTH - crop_width)
      amt_add_right = GENERATED_PIC_WIDTH - crop_width - amt_add_left
      rand_loc[0] -= amt_add_left
      rand_loc[2] += amt_add_right
    waldo_im = im.crop(rand_loc)
    mask_im = Image.open(TRAINING_PATH + name + '_mask.png')
    waldo_mask_im = mask_im.crop(rand_loc)
    images_np.append(np.asarray(waldo_im, dtype=np.float32))
    mask_images_np.append(np.divide(np.asarray(waldo_mask_im, dtype=np.float32), 255.0))
    
    #images without waldo
    for i in range(num_without_waldo_per_waldo):
      while 1:
        x1, y1 = random.randint(0, width - GENERATED_PIC_WIDTH), random.randint(0, height - GENERATED_PIC_HEIGHT)
        x2, y2 = x1 + GENERATED_PIC_WIDTH, y1 + GENERATED_PIC_HEIGHT
        for loc in locs:
          if is_overlapping([x1, y1, x2, y2], loc):
            break
        else:
          break
      images_np.append(np.asarray(im.crop([x1, y1, x2, y2]), dtype=np.float32))
      mask_images_np.append(np.divide(np.asarray(mask_im.crop([x1, y1, x2, y2]), dtype=np.float32), 255.0))
  return np.array(images_np), np.array(mask_images_np)


def main():
  X = tf.placeholder(tf.float32)
  Y = tf.placeholder(tf.float32)

  x = tf.reshape(tf.cast(X, tf.float32), [-1, 224, 224, 3])
  y = tf.reshape(tf.cast(Y, tf.float32), [-1, 224, 224, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=x,
      filters=32,
      kernel_size=3,
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=32,
      kernel_size=3,
      padding="same",
      activation=tf.nn.relu)
  # pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Convolutional Layer #3
  conv3 = tf.layers.conv2d(
      inputs=conv2,
      filters=64,
      kernel_size=3,
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #4
  conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=64,
      kernel_size=3,
      padding="same",
      activation=None)

  # Convolutional Layer #5
  conv5 = tf.layers.conv2d(
      inputs=conv4,
      filters=128,
      kernel_size=3,
      padding="same",
      activation=tf.nn.relu)

  # Convolutional Layer #6
  conv6 = tf.layers.conv2d(
      inputs=conv5,
      filters=128,
      kernel_size=3,
      padding="same",
      activation=None)

  # Convolutional Layer #7 (1x1 filters to map to 1 channel)
  logits = tf.layers.conv2d(
      inputs=conv6,
      filters=1,
      kernel_size=1,
      activation=None
      )

  prediction = tf.greater(tf.sigmoid(logits), 0.5)

  correct_prediction = tf.equal(prediction, tf.cast(y, bool))

  ####
  #pixel accuracy and sensitivity
  pixel_accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32), axis=[1,2,3]) / (224*224)
  zero = tf.constant(0, tf.float32)
  condition = tf.not_equal(tf.reduce_sum(y, axis=[1,2,3]), zero)
  indices = tf.where(condition)
  pixel_true_positives = tf.gather_nd(tf.reduce_sum(tf.cast(tf.logical_and(prediction, tf.cast(y, bool)), tf.float32), axis=[1,2,3]), indices)
  pixel_num_true = tf.gather_nd(tf.reduce_sum(y, axis=[1,2,3]), indices)
  pixel_sensitivity = tf.div(pixel_true_positives, pixel_num_true)


  loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=y, logits=logits, pos_weight=30))
  optimizer = tf.train.AdamOptimizer(1e-4)
  train = optimizer.minimize(loss)

  with tf.Session() as sess:
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter("/tmp/log/...", sess.graph)
    sess.run(tf.global_variables_initializer())

    for i in range(1, 51):
      images, mask = generate_train_data(4, 0) # 2 pics with waldo and 1 no waldos per waldo = 4 images per batch
      sess.run(train, feed_dict={X: images, Y: mask})
      metrics = sess.run([loss, pixel_accuracy, pixel_sensitivity], feed_dict={X: images, Y: mask})
      print("Step {}, Loss {}, Accuracy {}, Sensitivity {}".format(i, metrics[0], metrics[1], metrics[2]))
      if i % 10 == 0:
        saver.save(sess, './find_waldo_model/model', global_step=i)
    writer.close()

if __name__ == '__main__':
  main()
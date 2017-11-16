# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
import os
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
#import c3d_model
import resnet_c3d
import math
import numpy as np
import utils

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 1
#flags.DEFINE_float('learning_rate', 0.0, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 5000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 2, 'Batch size.')
FLAGS = flags.FLAGS
MOVING_AVERAGE_DECAY = 0.9999
model_save_dir = './models'



def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                        resnet_c3d.NUM_FRAMES_PER_CLIP,
                                                        resnet_c3d.CROP_SIZE,
                                                        resnet_c3d.CROP_SIZE,
                                                        resnet_c3d.CHANNELS))

  labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
  return images_placeholder, labels_placeholder

def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def tower_loss(name_scope, logit, labels):
  cross_entropy_mean = tf.reduce_mean(
                  tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logit)
                  )
  tf.summary.scalar(
                  name_scope + '_cross_entropy',
                  cross_entropy_mean
                  )
  weight_decay_loss = tf.get_collection('weightdecay_losses')
  tf.summary.scalar(name_scope + '_weight_decay_loss', tf.reduce_mean(weight_decay_loss) )

  # Calculate the total loss for the current tower.
  total_loss = cross_entropy_mean + weight_decay_loss 
  tf.summary.scalar(name_scope + '_total_loss', tf.reduce_mean(total_loss) )
  return total_loss

def tower_acc(logit, labels):
  correct_pred = tf.equal(tf.argmax(logit, 1), labels)
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  return accuracy

def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, wd):
  var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
  if wd is not None:
    weight_decay = tf.nn.l2_loss(var)*wd
    tf.add_to_collection('weightdecay_losses', weight_decay)
  return var

def run_training():
  # Get the sets of images and labels for training, validation, and
  # Tell TensorFlow that the model will be built into the default Graph.

  # Create model directory
  if not os.path.exists(model_save_dir):
      os.makedirs(model_save_dir)
  use_pretrained_model = False
  model_filename = "./sports1m_finetuning_ucf101.model"

  with tf.Graph().as_default():
    global_step = tf.get_variable(
                    'global_step',
                    [],
                    initializer=tf.constant_initializer(0),
                    trainable=False
                    )
    images_placeholder, labels_placeholder = placeholder_inputs(
                    FLAGS.batch_size * gpu_num
                    )
    tower_grads1 = []
    tower_grads2 = []
    logits = []
    opt_stable = tf.train.AdamOptimizer(1e-4)
    opt_finetuning = tf.train.AdamOptimizer(1e-3)
    
    for gpu_index in range(0, gpu_num):
      with tf.device('/gpu:%d' % gpu_index):
        
        
        #varlist2 = [ weights['out'],biases['out'] ]
        #varlist1 =  list(set(list(weights.values()) + list(biases.values())) - set(varlist2))
        logit = resnet_c3d.inference_c3d(
                        images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:,:,:,:])
                      #  0.5,
                      #  FLAGS.batch_size,
                        #weights,
                       # biases
                      #  )

        varlist2 = []
        weights = []
        biases = []
        for var in tf.trainable_variables():
            print(var.op.name)
            if var.op.name.find(r"weight") > 0:
                varlist2.append(var)
                weights.append(var)
            if var.op.name.find(r"bias") > 0 :
                varlist2.append(var)
                biases.append(var)
        
        print("len(varlist2)")
        print(len(varlist2))
        loss_name_scope = ('gpud_%d_loss' % gpu_index)
        loss = tower_loss(
                        loss_name_scope,
                        logit,
                        labels_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size]
                        )
        #grads1 = opt_stable.compute_gradients(loss, varlist1)
        grads2 = opt_finetuning.compute_gradients(loss, varlist2)
        #tower_grads1.append(grads1)
        tower_grads2.append(grads2)
        logits.append(logit)
    logits = tf.concat(logits,0)
    accuracy = tower_acc(logits, labels_placeholder)
    tf.summary.scalar('accuracy', accuracy)
    grads1 = average_gradients(tower_grads1)
    #grads2 = average_gradients(tower_grads2)
    #apply_gradient_op1 = opt_stable.apply_gradients(grads1)
    apply_gradient_op2 = opt_finetuning.apply_gradients(grads2, global_step=global_step)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    #train_op = tf.group(apply_gradient_op1, apply_gradient_op2, variables_averages_op)
    train_op = tf.group(apply_gradient_op2, variables_averages_op)
    null_op = tf.no_op()

    # Create a saver for writing training checkpoints.
    #saver = tf.train.Saver(list(weights.values()) + list(biases.values()))
    saver = tf.train.Saver(weights+biases)
    init = tf.global_variables_initializer()

    # Create a session for running Ops on the Graph.
    sess = tf.Session(
                    config=tf.ConfigProto(allow_soft_placement=True)
                    )
    sess.run(init)
    if os.path.isfile(model_filename) and use_pretrained_model:
      saver.restore(sess, model_filename)

    # Create summary writter
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./visual_logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('./visual_logs/test', sess.graph)
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      train_images, train_labels, _, _, _ = input_data.read_clip_and_label(
                      filename='./list/train.list',
                      batch_size=FLAGS.batch_size * gpu_num,
                      num_frames_per_clip=resnet_c3d.NUM_FRAMES_PER_CLIP,
                      crop_size=resnet_c3d.CROP_SIZE,
                      shuffle=True
                      )
      print("train_images.shape")
      print(train_images.shape)
      sess.run(train_op, feed_dict={
                      images_placeholder: train_images,
                      labels_placeholder: train_labels
                      })
      duration = time.time() - start_time
      print('Step %d: %.3f sec' % (step, duration))

      # Save a checkpoint and evaluate the model periodically.
      if (step) % 10 == 0 or (step + 1) == FLAGS.max_steps:
        saver.save(sess, os.path.join(model_save_dir, 'c3d_ucf_model'), global_step=step)
        print('Training Data Eval:')
        summary, acc = sess.run(
                        [merged, accuracy],
                        feed_dict={images_placeholder: train_images,
                            labels_placeholder: train_labels
                            })
        print ("accuracy: " + "{:.5f}".format(acc))
        train_writer.add_summary(summary, step)
        print('Validation Data Eval:')
        val_images, val_labels, _, _, _ = input_data.read_clip_and_label(
                        filename='./list/test.list',
                        batch_size=FLAGS.batch_size * gpu_num,
                        num_frames_per_clip=resnet_c3d.NUM_FRAMES_PER_CLIP,
                        crop_size=resnet_c3d.CROP_SIZE,
                        shuffle=True
                        )
        summary, acc = sess.run(
                        [merged, accuracy],
                        feed_dict={
                                        images_placeholder: val_images,
                                        labels_placeholder: val_labels
                                        })
        print ("accuracy: " + "{:.5f}".format(acc))
        test_writer.add_summary(summary, step)
  print("done")

def main(_):
  run_training()

if __name__ == '__main__':
  tf.app.run()

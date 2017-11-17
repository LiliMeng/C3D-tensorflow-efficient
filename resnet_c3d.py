"""
Authors:
Lili Meng, Nov 16th, 2017
University of British Columbia and Xtract Technologies
"""

from utils import *
import tensorflow as tf

# The UCF-101 dataset has 101 classes
NUM_CLASSES = 101

# Images are cropped to (CROP_SIZE, CROP_SIZE)
# The original video frames are first resized into 128x171 (roughly half resolution of the UCF101 frames)
# Then the jittering is used, random crop to 3x16x112x112
CROP_SIZE = 112
CHANNELS = 3

# Number of frames per video clip
NUM_FRAMES_PER_CLIP = 16

kernel_depth = 3
kernel_height = 3
kernel_width = 3

output_channel = [16, 32, 64]
num_units = 3
num_blocks = [3, 3, 3]

batch_size = 4
"-----------------------------------------------------------------------------------------------------------------------"


def resent_c3d_basic_block(Y0, input_channel, output_channel, scope_name):
    """ResNet C3D basic block"""
    with tf.variable_scope(scope_name) as scope:

        assert(input_channel == output_channel)
        # 3D convolution kernel
        K1 = weight_variable([kernel_depth, kernel_height, kernel_width,
                              input_channel, output_channel], 'weight_K1')

        K2 = weight_variable([kernel_depth, kernel_height, kernel_width,
                              output_channel, output_channel], 'weight_K2')

        Y = conv3d('conv1', Y0, K1)
        # Y = diagnal_3dconv('conv1', Y0, input_channel)
        # add bias
        Y = Y + bias_variable([output_channel], 'bias_b1')

        Y = tf.nn.relu(Y, 'relu_r1')

        Y = conv3d('conv2', Y, K2)

        Y = Y + bias_variable([output_channel], 'bias_2')

        scope.reuse_variables()

        return Y0 + Y


def resent_c3d_connective_block(Y, input_channel, output_channel, stride, scope_name):

    with tf.variable_scope(scope_name) as scope:
        K_conn = weight_variable([kernel_depth, kernel_height, kernel_width,
                                  input_channel, output_channel], 'weight_K_connecitve')

        Y = conv3d('connect_conv', Y, K_conn)

        Y = Y + bias_variable([output_channel], 'bias_connect')

        Y = tf.nn.relu(Y, 'relu_conv')

        Y = tf.nn.avg_pool3d(Y, ksize=[1, kernel_depth, 2, 2, 1], strides=[
                             1, kernel_depth, 2, 2, 1], padding='SAME', name="avg_pool_conn")

        scope.reuse_variables()

        return Y


def resnet_c3d_unit(Y, num_blocks, input_channel, output_channel, stride, scope_name):
    """ResNet C3D unit"""
    # The last basic block is different from previous ones
    # It has different input/output channels and the stride is not 1

    with tf.variable_scope(scope_name) as scope:
        for i in range(num_blocks - 1):
            Y = resent_c3d_basic_block(Y, input_channel=input_channel,
                                       output_channel=input_channel, scope_name="block" + str(i))
        Y = resent_c3d_connective_block(Y, input_channel=input_channel, output_channel=output_channel,
                                        stride=stride, scope_name="block" + str(num_blocks - 1))

        return Y


def inference_c3d(Y):

    reuse = None

    with tf.variable_scope('resnet_c3d', reuse=reuse):

            # The input Y shape (2, 16, 112, 112, 3)

        img_channel = Y.shape[4]

        # hahahaha for the 3D filter, we are using the (3x3x3) filter, as it's the best mentioned in
        # the The Fig. 2 and Section 3.2 in this C3D paper  https://arxiv.org/pdf/1412.0767.pdf
        # Initial convolution -- open the image to more channels
        with tf.variable_scope('unit_initial', reuse=reuse) as scope:
            K0 = weight_variable([kernel_depth, kernel_height, kernel_width,
                                  img_channel, output_channel[0]], 'weight_K')

            Y = conv3d('unit_initial', Y, K0)
            # add bias
            Y = Y + bias_variable([output_channel[0]], 'bias_variable')

            for i in range(num_units - 1):
                Y = resnet_c3d_unit(Y, num_blocks[i],
                                    input_channel=output_channel[i],
                                    output_channel=output_channel[i + 1],
                                    stride=2,
                                    scope_name='unit' + str(i))

            # The last unit has the same input_channel and output_channel,
            # and stride=1
            Y = resnet_c3d_unit(Y, num_blocks[-1],
                                input_channel=output_channel[-1],
                                output_channel=output_channel[-1],
                                stride=1,
                                scope_name='unit' + str(num_units - 1))

           
            # average pooling layer
            Y = tf.reduce_mean(Y, [2, 3])


            Y = tf.reshape(Y, [batch_size, -1])

            # Fully connected layer weight
            FC_W = weight_variable(
                [Y.shape[1], NUM_CLASSES], 'weight_FC')
            FC_b = bias_variable([NUM_CLASSES], 'bias_FC')

            scope.reuse_variables()

            logits = tf.nn.xw_plus_b(Y, FC_W, FC_b)

            return logits

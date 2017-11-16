import tensorflow as tf

def bias_variable(shape, name):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name, dtype=tf.float32, initializer=initial)

def weight_variable(shape, scope_name):
    with tf.name_scope(scope_name):
        return tf.get_variable(scope_name, shape, initializer=tf.truncated_normal_initializer(stddev=0.1))


def conv3d(scope_name, l_input, w):
    with tf.name_scope(scope_name):
        return tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME')

def conv2d_transpose(x,W):
    "conv2d_transpose returns a transposed 2d convolution layer with full stride"
    return tf.nn.conv2d_transpose(x, W,
                                  output_shape=tf.concat(
                                    [tf.shape(x)[0:3],[tf.shape(W)[2]]], axis=0),
                                  strides=[1,1,1,1],
                                  padding='SAME')

def conv3d_transpose(x,W):
    return tf.nn.conv3d_transpose(x, W,
                                  output_shape=tf.concat(
                                    [tf.shape(x)[0:4],[tf.shape(W)[3]]], axis=0),
                                    strides =[1,1,1,1,1],
                                    padding = 'SAME')


def diagnal_3dconv_transpose(scope_name, Y, channel):
    with tf.variable_scope(scope_name, reuse=False):

       # Step 1: The diffusion term

        K_d = weight_variable([3,3,3,channel,channel], 'K_d'+scope_name)

        Z_diffusion = conv3d_transpose(Y, K_d)
        
         # # Step 2: The reaction term

        K_r = weight_variable(
            [1, 1, 1, channel, channel], 'weight_side1' + scope_name)

        Z_reaction = conv3d('Z_reaction',Y, K_r)

        # Step 3: add the diffusion and reaction together
        Z_output = Z_diffusion + Z_reaction

        return Z_output


def diagnal_3dconv(scope_name, Y, channel):

    with tf.variable_scope(scope_name, reuse=False):

         # Step 1: The diffusion term
        Z_diffusion = []

        # # The diagnal will be filter [3,3,1,1]
       
        for i in range(channel):
            K_d = weight_variable([3, 3, 3, 1, 1], 'weight_diagnal' + str(i+1))
            Y_one_channel = Y[:, :, :, :, i:i + 1]
            singleYK = conv3d(scope_name, Y_one_channel,K_d)
            Z_diffusion.append(singleYK)

        Z_tmp = tf.stack(Z_diffusion)      
        Z_diffusion = tf.transpose(tf.squeeze(Z_tmp, 5), [1, 2, 3, 4, 0])
      

        # # Step 2: The reaction term

        K_r = weight_variable(
            [1, 1, 1, channel, channel], 'weight_side1' + scope_name)

        Z_reaction = conv3d('Z_reaction',Y, K_r)

        # Step 3: add the diffusion and reaction together
        Z_output = Z_diffusion + Z_reaction

        return Z_output


def bias_variable(shape, name):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name, dtype=tf.float32, initializer=initial)


   
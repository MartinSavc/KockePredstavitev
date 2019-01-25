import tensorflow as tf

def resnet_block(inputs, filters, kernel_size, training, name=None):
    name_bn1 = None
    name_relu1 = None
    name_conv1 = None
    name_bn2 = None
    name_relu2 = None
    name_conv2 = None
    N = inputs.shape[-1]
    
    if name is not None:
        name_bn1 = name+'_bn1'
        name_relu1 = name+'_relu1'
        name_conv1 = name+'_conv1'
        name_bn2 = name+'_bn2'
        name_relu2 = name+'_relu2'
        name_conv2 = name+'_conv2'
        
    bn1 = tf.layers.batch_normalization(inputs=inputs, 
                                        training=training, 
                                        name=name_bn1)
    relu1 = tf.nn.relu(bn1, 
                       name=name_relu1)
    conv1 = tf.layers.conv2d(inputs=relu1, 
                             filters=filters,
                             kernel_size=kernel_size,
                             padding='same',
                             name=name_conv1)
    
    bn2 = tf.layers.batch_normalization(inputs=conv1, 
                                        training=training, 
                                        name=name_bn2)
    relu2 = tf.nn.relu(bn2, 
                       name=name_relu2)
    conv2 = tf.layers.conv2d(inputs=relu2,
                             filters=filters,
                             kernel_size=kernel_size,
                             padding='same',
                             name=name_conv2)
    return tf.add(inputs, conv2, name=name)

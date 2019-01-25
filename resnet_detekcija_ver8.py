import os
import numpy as np
import tensorflow as tf

from resnet_block import resnet_block

def conv_net_model(features, labels, mode):
    # Input Layer
    N, H, W, C = features['x'].shape
    input_layer = tf.reshape(features['x'], [-1, H, W, C])

    # are we training?
    training = mode==tf.estimator.ModeKeys.TRAIN

    kernel_size = (5, 5)
    layer_count = 5 # resnet layer contains 2 convolution layers
    filters = 8
    # receptive field -  5+10*(2+2) = 45
    # 5+10*4 = 45
    # invalid edge - (45-1)/2 = 22 ~ 25
    E = 25

    prev_layer = tf.layers.conv2d(inputs=input_layer,
                                  filters=filters,
                                  kernel_size=kernel_size,
                                  padding='same',
                                  name='conv_1')

    resnet_layers_dict = {}
    for l in range(layer_count):
        res_layer = resnet_block(inputs=prev_layer, 
        		     training=training,
        		     kernel_size=kernel_size,
        		     filters=filters,
        		     name='res_layer{:d}'.format(l+1))
        resnet_layers_dict['res_layer{:d}'.format(l+1)] = res_layer
        prev_layer = res_layer
    relu_top = tf.nn.relu(res_layer, 
    		      name='relu_top')
    
    output_map = tf.layers.conv2d(inputs=relu_top,
                                  filters=2,
                                  kernel_size=(1, 1),
                                  name='output_logits')
        
    predictions = {
        # Generate prediction map
        'map': tf.nn.softmax(output_map, axis=3)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions['input'] = input_layer
        predictions.update(resnet_layers_dict)
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    target_map = tf.reshape(labels, [-1, H, W, 2])
    target_probs = tf.reshape(target_map[:, E:-E, E:-E, :], (-1, 2))
    logits = tf.reshape(output_map[:, E:-E, E:-E, :], (-1, 2))

    # calculate weights to fix uneven class representation
    w1 = tf.reduce_mean(target_probs)
    class_weights = tf.stack([w1, 1-w1], axis=0, name='class_weights')
    weights = tf.gather(class_weights, tf.argmax(target_probs, axis=1))

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=target_probs, logits=logits, weights=weights)


    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.image('input_summary', input_layer)
        #tf.summary.image('weights_summary', tf.reshape(weights, [-1, H, W, 1]))
        tf.summary.image('target_summary', target_map[:, :, :, 1:2])
        tf.summary.image('output_summary', predictions['map'][:, :, :, 1:2])

        global_step = tf.train.get_global_step()
        learning_rate = tf.train.exponential_decay(0.01,
                                                   global_step,
                                                   10000,
                                                   0.8,
                                                   staircase=True,
                                                   name='learning_rate')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    #(for EVAL mode,  the only remaining mode)
    # Add evaluation metrics 
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=tf.argmax(target_map[:, E:-E, E:-E, :], axis=3), 
            predictions=tf.argmax(predictions['map'][:, E:-E, E:-E, :], axis=3))
    }

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

if __name__ == '__main__':
    import load_and_prep_data as load_data
    model_dir = 'resnet_ver8_F_8_L_5_K_5_5/'
    try:
        os.mkdir(model_dir)
    except:
        pass

    tmp_data_file = os.path.join(model_dir, 'tmp_data.npz')
    if os.path.isfile(tmp_data_file):
        data = np.load(tmp_data_file)
        imgs = data['imgs']
        masks = data['masks']
        data = None
    else:
        data = list(load_data.load_resample_and_prep_data('./data_2018_09_11/', win_size=128, resamples=16*3))
        data += list(load_data.load_resample_and_prep_data('./data_2018_09_13/', win_size=128, resamples=16*3))
        imgs = np.array([i for i, _ in data])
        masks = np.array([m for _, m in data])
        data = None
        np.savez(tmp_data_file, imgs=imgs, masks=masks)

    print('sample array size: {:}'.format(imgs.shape))
    imgs[imgs<0] = 0
    imgs[imgs>1] = 1
    imgs = imgs.mean(3, keepdims=True)
    masks[masks<1e-5] = 1e-5
    masks[masks>1-1e-5] = 1-1e-5

    imgs = imgs.reshape(imgs.shape[0], imgs.shape[1]//2, 2, imgs.shape[2]//2, 2, -1).mean(axis=(2, 4))
    masks = masks.reshape(masks.shape[0], masks.shape[1]//2, 2, masks.shape[2]//2, 2, -1).mean(axis=(2, 4))

    target_maps = np.zeros(masks.shape+(2,), dtype=np.float32)
    target_maps[..., 0] = 1-masks
    target_maps[..., 1] = masks

    test_sample_mask = np.zeros(imgs.shape[0], dtype=np.bool8)
    test_sample_mask[-(6*16*3):] = True
    #test_sample_mask[1:] = True

    
    imgs_train = imgs[test_sample_mask==False]
    target_maps_train = target_maps[test_sample_mask==False]

    imgs_test = imgs[test_sample_mask]
    target_maps_test = target_maps[test_sample_mask]


    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': imgs_train},
            y=target_maps_train,
            batch_size=128,
            num_epochs=None,
            shuffle=True)


    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': imgs_test},
            y=target_maps_test,
            num_epochs=1,
            shuffle=False)

    cnn_model_fn = conv_net_model
    tf_estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=model_dir)

    tensors_to_log = {}#'probabilities': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

    for n in range(200):
        tf_estimator.train(
                input_fn=train_input_fn,
                steps=1000,
                hooks=[logging_hook])

        eval_results = tf_estimator.evaluate(input_fn=eval_input_fn)



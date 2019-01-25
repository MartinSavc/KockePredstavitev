import tensorflow as tf
import matplotlib.pyplot as pyplot

from resnet_block import resnet_block
'''***'''
def simple_resnet_model_gen(input_shape, F, N_res_layers, N_outputs):
    if len(input_shape) == 3:
        H, W, C = input_shape
    elif len(input_shape) == 2:
        H, W = input_shape
        C = 1
    else:
        raise Exception('invalid input_shape')

    dense_dim = ((H-8)//4+1)*((W-8)//4+1)

    def resnet_model_template(features, labels, mode):
        # Input Layer
        input_layer = tf.reshape(features["x"], [-1, H, W, C])
        
        # are we training?
        training = mode==tf.estimator.ModeKeys.TRAIN
        
        # HxWxF
        conv1 = tf.layers.conv2d(inputs=input_layer,
                                 filters=F,
                                 kernel_size=[1, 1],
                                 padding='same',
                                 activation=tf.nn.relu,
                                 name='conv1')
        
        res_layers_dict = {}
        prev_layer = conv1
        for n in range(N_res_layers):
            res_layer = resnet_block(inputs=prev_layer, 
                                     training=training,
                                     kernel_size=[3, 3],
                                     filters=F,
                                     name=f'res_layer{n+1:d}')
            res_layers_dict[f'res_layer{n+1:d}'] = res_layer
            prev_layer = res_layer
        
        #HxWxF
        #
        relu_top = tf.nn.relu(res_layer, 
                              name='relu_top')
        conv_top = tf.layers.conv2d(inputs=prev_layer,
                                 filters=1,
                                 kernel_size=[8, 8],
                                 strides=[4, 4],
                                 padding="valid",
                                 activation=tf.nn.relu,
                                 name='conv_top')

        conv_top_flat = tf.reshape(conv_top, [-1, dense_dim], name='conv_top_flat')
        logits_out = tf.layers.dense(inputs=conv_top_flat, units=N_outputs, name='logits_out')

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits_out, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits_out, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions['input'] = input_layer
            predictions.update(res_layers_dict)
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits_out)

        
        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_global_step()
            learning_rate = tf.train.exponential_decay(0.001,
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
            "accuracy": tf.metrics.accuracy(
                labels=labels, 
                predictions=predictions["classes"])
        }
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    return resnet_model_template
'''***'''
def conv_resnet_model_gen(F, N_res_layers, kernel_size=(3, 3)):

    def resnet_model_template(features, labels, mode):
        # Input Layer
        N, H, W, C = features['x'].shape
        input_layer = tf.reshape(features['x'], [-1, H, W, C])
        
        # are we training?
        training = mode==tf.estimator.ModeKeys.TRAIN
        
        # HxWxF
        conv1 = tf.layers.conv2d(inputs=input_layer,
                                 filters=F,
                                 kernel_size=kernel_size,
                                 padding='same',
                                 activation=tf.nn.relu,
                                 name='conv1')
        
        res_layers_dict = {}
        prev_layer = conv1
        for n in range(N_res_layers):
            res_layer = resnet_block(inputs=prev_layer, 
                                     training=training,
                                     kernel_size=kernel_size,
                                     filters=F,
                                     name=f'res_layer{n+1:d}')
            res_layers_dict[f'res_layer{n+1:d}'] = res_layer
            prev_layer = res_layer
        
        #HxWxF
        #
        relu_top = tf.nn.relu(res_layer, 
                              name='relu_top')
        conv_top = tf.layers.conv2d(inputs=prev_layer,
                                 filters=1,
                                 kernel_size=(1, 1),
                                 padding='same',
                                 activation=tf.sigmoid,
                                 name='conv_top')

        predictions = {
            # Generate prediction map
            'map': conv_top,
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions['input'] = input_layer
            predictions.update(res_layers_dict)
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        output_map = tf.reshape(labels, [-1, H, W, 1])
        # Calculate Loss (for both TRAIN and EVAL modes)
        #sig_cross_ent_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=output_map, logits=conv_top)
        #loss = tf.losses.compute_weighted_loss(sig_cross_ent_loss)
        loss = tf.losses.absolute_difference(labels=output_map, predictions=conv_top)

        tf.summary.image('input_summary', input_layer)
        tf.summary.image('target_summary', output_map)
        tf.summary.image('output_summary', predictions['map'])

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_global_step()
            learning_rate = tf.train.exponential_decay(0.001,
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
            'accuracy': tf.metrics.mean_absolute_error(
                labels=output_map, 
                predictions=predictions['map'])
        }

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    return resnet_model_template

'''***'''
if __name__ == '__main__':
    import load_and_prep_dice_imgs_2 as data
    (train_input_fn, eval_input_fn,
     labels_train, imgs_train,
     labels_test, imgs_test) = data.load_and_prep()
    
    '''***'''
    N_res_layers = 8
    F_res_chns = 16
    # create estimator
    resnet_model_fn = simple_resnet_model_gen(input_shape=[32,32], F=F_res_chns, N_res_layers=N_res_layers, N_outputs=7)
    resnet_estimator = tf.estimator.Estimator(
        model_fn=resnet_model_fn, model_dir=f'./dice_2_model_resnet_pa_F_{F_res_chns:d}_N_{N_res_layers:d}')


    '''***'''
    # train estimator, with intermitent evaluation
    tensors_to_log = {}#"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)
    for n in range(1):
        resnet_estimator.train(
            input_fn=train_input_fn,
            steps=100,
            hooks=[logging_hook])

        eval_results = resnet_estimator.evaluate(input_fn=eval_input_fn)

    '''***'''
    # display some results  
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": imgs_test[:10]}, shuffle=False)
    r = resnet_estimator.predict(pred_input_fn)

    for v in r:

        im_in = v['input']
        class_prob = v['probabilities']

        res_layers = [v[f'res_layer{n+1:d}'] for n in range(N_res_layers)]
        
        pyplot.figure()
        pyplot.imshow(im_in[:, :, 0])
        
        pyplot.figure()
        pyplot.bar(np.arange(0, 7), class_prob, width=0.8)
        
        for im_array in res_layers:
                       
        
            N = im_array.shape[2]
            #fig,ax = pyplot.subplots(2,4)
            fig, ax = pyplot.subplots(int(N**0.5), int(N/int(N**0.5)+1))
            ax = ax.ravel()

            #M = np.abs(conv1_im).max()

            for n in range(0, im_array.shape[2]):
                ax[n].imshow(im_array[:,:,n])#, vmin=-M, vmax=M)
                ax[n].set_axis_off()
            for n in range(im_array.shape[2], ax.shape[0]):
                ax[n].set_axis_off()

            
    pyplot.show()
    '''***'''

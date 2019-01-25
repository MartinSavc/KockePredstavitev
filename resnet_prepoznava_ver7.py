import os
import numpy as np
import tensorflow as tf
import scipy.signal 

from resnet_block import resnet_block

class gen_convnet_model():
    def __init__(self, layer_count_list, filters_list, kernel_size_list, dense_layer_size, classes):
        self.L = layer_count_list
        self.F = filters_list
        self.K = kernel_size_list
        self.D = dense_layer_size
        self.classes = classes

    def __call__(self, features, labels, mode):
        # Input Layer
        input_layer = features['x']
        N, H, W, C = input_layer.shape

        # are we training?
        training = mode==tf.estimator.ModeKeys.TRAIN

        conv_layers_dict = {}
        prev_layer = input_layer
        for n in range(len(self.L)):
            filts = self.F[n]
            kern_size = self.K[n]
            layer_count = self.L[n]
  
            prev_layer = tf.layers.conv2d(inputs=prev_layer,
                                          filters=filts,
					  kernel_size=(1, 1),
                                          padding='same',
                                          name='chn_tran_{:d}_{:d}'.format(n,n+1))
            conv_layers_dict['chn_tran_{:d}_{:d}'.format(n,n+1)] = prev_layer

            for l in range(layer_count):
                res_layer = resnet_block(inputs=prev_layer, 
                                 training=training,
                                 kernel_size=kern_size,
                                 filters=filts,
                                 name='res_layer_{:d}_{:d}'.format(n+1, l+1))
                conv_layers_dict['res_layer_{:d}_{:d}'.format(n+1, l+1)] = res_layer
                prev_layer = res_layer
            prev_layer = tf.nn.pool(input=prev_layer,
                                    window_shape=(3, 3),
                                    pooling_type='MAX',
                                    padding='SAME',
                                    strides=(2, 2),
                                    name='pool_layer_{:}'.format(n+1))

        relu_top = tf.nn.relu(prev_layer, 
                              name='relu_top')

        relu_top_flat = tf.layers.flatten(relu_top)
        dl_1 = tf.layers.dense(inputs=relu_top_flat,
                               units=self.D,
                               name='dl_1')
        dl_1_relu = tf.nn.relu(dl_1, name='dl_1_relu')
        dl_2 = tf.layers.dense(inputs=dl_1_relu,
                        units=self.classes, 
                        name='dl_2')
        logits_out = dl_2

        predictions = {
                     # Generate prediction map
                     'probabilities': tf.nn.softmax(logits_out, axis=1),
		     'label': tf.argmax(logits_out, axis=1),
                     'feature': dl_1_relu
            }

        if mode == tf.estimator.ModeKeys.PREDICT:
            #predictions['input'] = input_layer
            #predictions['logits_out'] = logits_out
            #predictions.update(conv_layers_dict)
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits_out)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.image('input_summary', input_layer)
            for n in range(len(self.L)):
                l = self.L[n]
                conv_layer = conv_layers_dict['res_layer_{:d}_{:d}'.format(n+1, l)]
                _, _, W_, _ = conv_layer.shape
                tf.summary.image('res_layer_{:d}_{:}_summary'.format(n+1, l),
                                  tf.reshape(tf.transpose(conv_layer, (0, 3, 1, 2)), (N, -1, W_, 1)))
            for f in range(self.F[-1]):
                tf.summary.image('relu_top_F_{:}'.format(f+1), relu_top[:, :, :, f:f+1])
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
                train_op = optimizer.minimize(loss=loss,
                                              global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        #(for EVAL mode,  the only remaining mode)
        # Add evaluation metrics 
        eval_metric_ops = {
                           'accuracy': tf.metrics.accuracy(labels=labels, 
                                                           predictions=tf.argmax(logits_out, axis=1))
                          }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

if __name__ == '__main__':
    import load_and_prep_data as load_data
    F = (4,) 
    L = (3,)
    K = ((5, 5),)
    D = 8
    win_fun_call = scipy.signal.cosine
    #win_fun_call = scipy.signal.hann
    #win_fun_call = scipy.signal.boxcar
    model_dir = ('resnet_ver7','F_{:}_L_{:}_K_{:}_D_{:}_sinewin/'.format(F, L, K, D))
    try:
        os.mkdir(model_dir[0])
    except:
        print('WARNING: model directory already exists')
        pass

    sample_size=41
    
    tmp_data_file = os.path.join(model_dir[0], 'tmp_data.npz')
    if os.path.isfile(tmp_data_file):
        data = np.load(tmp_data_file)
        train_imgs = data['train_imgs']
        train_labels = data['train_labels']
        eval_imgs = data['eval_imgs']
        eval_labels = data['eval_labels']
    else:
        print('generating training samples')
        train_img_val_list = [(img, val) for img, _, val in \
                              load_data.load_resample_and_prep_dice_data_from_folders(\
                               ['./data_2018_09_11', './data_2018_09_13'],\
                                win_size=sample_size, resamples=16*3*5, trans_range=(-4, 4))] # 16 rotations, 3 scales, 5 translations on average
        print('generating evaluation samples')
        eval_img_val_list = [(img, val) for img, _, val in \
                          load_data.load_resample_and_prep_dice_data_from_folders(\
                               './data_2018_09_14',\
                                win_size=sample_size, resamples=10, trans_range=(-4, 4))]

        print('transforming samples to arrays')
        train_imgs = np.array([img for img,_ in train_img_val_list])
        train_labels = np.array([val for _,val in train_img_val_list])-1

        eval_imgs = np.array([img for img,_ in eval_img_val_list])
        eval_labels = np.array([val for _,val in eval_img_val_list])-1
        print('data shapes - training imgs {:}, evalution imgs {:}'.format(train_imgs.shape, eval_imgs.shape))

        del train_img_val_list
        del eval_img_val_list
        np.savez(tmp_data_file, train_imgs=train_imgs,
                                eval_imgs=eval_imgs,
                                train_labels=train_labels,
                                eval_labels=eval_labels)

    win_function = win_fun_call(sample_size).reshape(1, -1)
    #win_function = scipy.signal.hann(sample_size).reshape(1, -1)
    win_function = win_function*win_function.T
    win_function = win_function.reshape(1, sample_size, sample_size, 1)

    train_imgs*=win_function
    eval_imgs*=win_function

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_imgs},
            y=train_labels,
            batch_size=128,
            num_epochs=None,
            shuffle=True)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_imgs},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)

    #def train_input_fn():
    #    def gen_fun():
    #            yield {'x':img, 'y':[val-1]}
    #        
    #    ds = tf.data.Dataset.from_generator(gen_fun,
    #                                        {'x':tf.float32, 'y':tf.int32}, 
    #                                        {'x':tf.TensorShape([sample_size, sample_size, 3]), 'y':tf.TensorShape([1])}) 
    #    ds = ds.batch(128)
    #    ds = ds.shuffle(16*3*6*2)

    #    return ds


    #def eval_input_fn():
    #    def gen_fun():
    #        for img, _, val in load_data.load_resample_and_prep_dice_data_from_folders('./data_2018_09_14', win_size=sample_size, resamples=16*3):
    #            yield {'x':img, 'y':[val-1]}
    #        
    #    ds = tf.data.Dataset.from_generator(gen_fun,
    #                                        {'x':tf.float32, 'y':tf.int32}, 
    #                                        {'x':tf.TensorShape([sample_size, sample_size, 3]), 'y':tf.TensorShape([1])}) 
    #    ds = ds.batch(128)
    #    ds = ds.repeat(1)
    #    return ds


    cnn_model_fn = gen_convnet_model(L, F, K, D, 6)
    tf_estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=os.path.join(*model_dir))

    tensors_to_log = {}#'probabilities': 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

    for n in range(200):
        tf_estimator.train(
                input_fn=train_input_fn,
                steps=1000,
                hooks=[logging_hook])

        eval_results = tf_estimator.evaluate(input_fn=eval_input_fn)



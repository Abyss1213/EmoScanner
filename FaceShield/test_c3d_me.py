"""
test
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
from six.moves import xrange    # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
import c3d_model
import numpy as np

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 1
flags.DEFINE_integer('batch_size', 10 , 'Batch size.')
FLAGS = flags.FLAGS

EMOTIONS = ['happiness', 'disgust', 'repression', 'surprise','others']

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
                                                                                                                 c3d_model.NUM_FRAMES_PER_CLIP,
                                                                                                                 c3d_model.CROP_SIZE,
                                                                                                                 c3d_model.CROP_SIZE,
                                                                                                                 c3d_model.CHANNELS))
    labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
    return images_placeholder, labels_placeholder

def _variable_on_cpu(name, shape, initializer):
    #with tf.device('/cpu:%d' % cpu_id):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd):
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var) * wd
        tf.add_to_collection('losses', weight_decay)
    return var

def run_test(model_name, test_list_file, result_file='predict.txt', predict=False):
    tf.reset_default_graph()
    num_test_videos = len(list(open(test_list_file,'r')))
    #print("Number of test videos={}".format(num_test_videos))

    # Get the sets of images and labels for training, validation, and
    images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size * gpu_num)
    with tf.variable_scope('var_name') as var_scope:
        weights = {
                        'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00),
                        'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
                        'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
                        'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
                        'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
                        'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
                        'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
                        'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
                        'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
                        'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002),
                        'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.04, 0.005)
                        }
        biases = {
                        'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
                        'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
                        'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
                        'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
                        'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
                        'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
                        'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
                        'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),
                        'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0),
                        'bd2': _variable_with_weight_decay('bd2', [4096], 0.04, 0.0),
                        'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.04, 0.0),
                        }
    logits = []
    for gpu_index in range(0, gpu_num):
        with tf.device('/gpu:%d' % gpu_index):
            logit = c3d_model.inference_c3d(images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:,:,:,:], 0.6, FLAGS.batch_size, weights, biases)
            logits.append(logit)
    logits = tf.concat(logits,0)

    '''
    clf2.predict(logits)
    clf2.predict_proba(logits)
    '''
    
    norm_score = tf.nn.softmax(logits)
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)
    # Create a saver for writing training checkpoints.
    saver.restore(sess, model_name)
    # And then after everything is built, start the training loop.
    bufsize = 0
    write_file = open(result_file, "a+")
    write_file.write(model_name+'\n')
    next_start_pos = 0
    all_steps = int((num_test_videos - 1) / (FLAGS.batch_size * gpu_num) + 1)
    accuracy = 0.0
    false_4 = 0.0
    results_array = []
    result_array = []
    
    '''
    #读取Model
    with open('save/clf.pickle', 'rb') as f:
        clf2 = pickle.load(f)
        #测试读取后的Model
        print(clf2.predict(X[0:1]))
    '''
    for step in xrange(all_steps):
        # Fill a feed dictionary with the actual set of images and labels
        # for this particular training step.
        start_time = time.time()
        test_images, test_labels, next_start_pos, _, valid_len = \
                        input_data.read_clip_and_label(
                                        test_list_file,
                                        FLAGS.batch_size * gpu_num,
                                        start_pos=next_start_pos
                                        )
        predict_score = norm_score.eval(
                        session=sess,
                        feed_dict={images_placeholder: test_images}
                        )
        if(predict):
            for i in range(0, valid_len):
                results = []
                result = ""
                true_label = test_labels[i]
                top1_predicted_label = np.argmax(predict_score[i])
                
                for j in range(0,len(predict_score[i])):
                    #print('{}:{:.5f}'.format(EMOTIONS[j], predict_score[i][j]))
                    write_file.write('{}:{:.5f}\n'.format(EMOTIONS[j], predict_score[i][j]))
                    results.append(predict_score[i][j])
                        
                #print('result:{}'.format(EMOTIONS[top1_predicted_label]))
                result = EMOTIONS[top1_predicted_label]
                write_file.write('result:{}\n'.format(result))
                results_array.append(results)
                result_array.append(result)
                
            write_file.close()
            return results_array, result_array
        
        for i in range(0, valid_len):
            true_label = test_labels[i],
            top1_predicted_label = np.argmax(predict_score[i])
            # Write results: true label, class prob for true label, predicted label, class prob for predicted label
            write_file.write('{}, {}\n'.format(
                            true_label[0],
                            top1_predicted_label))
            if(true_label[0] == top1_predicted_label):
                    accuracy +=1
            elif(true_label[0]!=4 and top1_predicted_label==4):
                    false_4 +=1
    accuracy /= num_test_videos
    false_4 /= num_test_videos
    write_file.write(model_name + 'accuracy: '+str(accuracy) + '\n')
    #write_file.write('false to others: '+str(false_4)+ '\n')
    #write_file.write('accuracy plus false to others: '+str(accuracy + false_4)+ '\n\n')
    write_file.close()
    print(model_name + "done")

def main(_):
    #for k_num in range(0,5):
    #    for i in range(1,6):
    with open('model/models.txt') as f:
        models = list(f)
        for model in models:
            model = model.strip()
            k = model.replace('c3d_me_model','')[0]
            print(model,'list/valid_part_'+k+'.list')
            run_test('model/'+model,'list/valid_part_'+k+'.list')
            #run_test("model%d/c3d_me_model-%d"%(k_num,i*1000), 'list/valid_part_'+str(k_num)+'.list')

if __name__ == '__main__':
    tf.app.run()

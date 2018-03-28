import tensorflow as tf
import imagenet
import mobilenet_v2
import mobilenet_v1
import numpy as np
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
slim = tf.contrib.slim

#data_dir   = '/mnt/cephfs/labcv/kangyang/datasets/ImageNet/tensorflow'
#data_dir   = '/mnt/cephfs/labcv/public_datasets/ImageNet/tf/val/'
data_dir = '/mnt/cephfs/labcv/guicunbin/datasets/ImageNet/tf/has_shuffled'
#ckpt_to_restore = '/mnt/cephfs/labcv/kangyang/classification/imagenet/tensorflow/exp/init_5/model.ckpt-183518'
ckpt_to_restore = '/data00/kangyang/segmentation/deeplab/datasets/pascal_voc_seg/init_models/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt'

dataset = imagenet.ImageNetDataSet(data_dir, 'validation', True)
X_op, y_op  = dataset.make_batch(1)

#with tf.variable_scope('mobilenet_v2', reuse=False):
#    with slim.arg_scope(mobilenet_v2.training_scope(is_training=False, dropout_keep_prob=0.8)):
#        logits, end_points = mobilenet_v2.mobilenet(X_op, num_classes=1000, prediction_fn=slim.softmax)
#        label_acc   = tf.metrics.accuracy(tf.one_hot(y_op, depth=1000), logits)
with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=False)):
    logits, end_points = mobilenet_v1.mobilenet_v1(X_op, num_classes=1000)
    prob = slim.softmax(logits)
    label_acc   = tf.metrics.accuracy(tf.one_hot(y_op, depth=1000), logits)


correct_pred = 0.0
batch_total = 0.0
init_l = tf.local_variables_initializer()
with tf.Session() as sess:
    sess.run(init_l)
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_to_restore)
    for step_i in range(50000):
        [acc, lg, label, im, p] = sess.run([label_acc, logits, y_op, X_op, prob])
        print np.argmax(lg), label
        import ipdb; ipdb.set_trace()
        im = (im[0] * 0.5 + 0.5) * 255
        #im = (im[0] + 1.0) * 255.0 / 2.0
        cv2.imwrite('test.png', im.astype(np.uint8)[:,:,::-1])
        #correct_pred += acc
        #print acc


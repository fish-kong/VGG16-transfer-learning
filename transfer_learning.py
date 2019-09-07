# -*- coding: utf-8 -*-
"""
The VGG model and parameters are adopted from:
https://github.com/machrisaa/tensorflow-vgg
Learn more, visit my tutorial site: [莫烦Python](https://morvanzhou.github.io)
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.reset_default_graph()
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import pdb
class Vgg16:

    def __init__(self, vgg16_npy_path=None, restore_from=None):
        # 加载预训练模型
        try:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1',allow_pickle=True).item()
        except FileNotFoundError:
            print('Please download VGG16 parameters from here https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM\nOr from my Baidu Cloud: https://pan.baidu.com/s/1Spps1Wy0bvrQHH2IMkRfpg')

        self.tfx = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.tfy = tf.placeholder(tf.float32, [None, 10])
        
        '''
        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.tfx * 255.0)
        bgr = tf.concat(axis=3, values=[
            blue - self.vgg_mean[0],
            green - self.vgg_mean[1],
            red - self.vgg_mean[2],
        ])'''
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        bgr = self.tfx*255 - mean

        # 利用训练好的VGG参数 给前几层网络的参数赋值，就是迁移
        conv1_1 = self.conv_layer(bgr, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool(conv3_3, 'pool3')

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        pool4 = self.max_pool(conv4_3, 'pool4')

        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        pool5 = self.max_pool(conv5_3, 'pool5')

        # 设定自己的全连接层，适应自己的类别
       
        pool_shape = pool5.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        self.reshaped = tf.reshape(pool5, [-1, nodes])
        with tf.variable_scope('fc6'):
            self.fc1_weights = tf.get_variable("weight", [nodes, 256],
                                          initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.fc1_biases = tf.get_variable("bias", [256], initializer=tf.constant_initializer(0.01))
            self.fc1 = tf.nn.relu(tf.matmul(self.reshaped, self.fc1_weights) + self.fc1_biases)
            

        with tf.variable_scope('fc7'):
            self.fc2_weights = tf.get_variable("weight", [256, 10],
                                          initializer=tf.truncated_normal_initializer(stddev=0.01))               
            self.fc2_biases = tf.get_variable("bias", 10, initializer=tf.constant_initializer(0.01))
            self.out = tf.matmul(self.fc1, self.fc2_weights) + self.fc2_biases
        self.predict=tf.nn.softmax(self.out)
        
        self.sess = tf.Session()
        
        if restore_from:
            saver = tf.train.Saver()
            saver.restore(self.sess, restore_from)
        else:   # training graph
            self.loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.tfy,logits=self.out))
            #self.loss = tf.reduce_mean(-tf.reduce_sum( self.tfy * tf.log(tf.nn.softmax(self.out)),reduction_indices=[1]))
            self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss)
            self.sess.run(tf.global_variables_initializer())

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):   # CNN's filter is constant, NOT Variable that can be trained
            conv = tf.nn.conv2d(bottom, self.data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, self.data_dict[name][1]))
            return lout

    def train(self, x, y):
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.tfx: x, self.tfy: y})
        return loss

    def predict_label(self, x):
        out = self.sess.run(self.predict,feed_dict={self.tfx: x})
        
        return out

    def save(self, path='transfer_learn/'):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=False)
# In[]
def read_data(data_dir):
    datas = []
    fpaths = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        fpaths.append(fpath)
        image = Image.open(fpath)
        image=image.resize((224,224))
        data = np.array(image)/255.0
        datas.append(data)
        
    datas = np.array(datas)
    datas=datas.astype(np.float32)
    
    print("shape of datas: {}".format(datas.shape))
    return datas
def read_label(filename):
    label=[]
    file=open(filename,'r')
    for line in file.readlines():
        b=line.split('\t')
        label.append(int(b[1][0]))
    label=np.array(label)
    
    labels=np.zeros([len(label),10],dtype='float32')
    for e in range(len(label)):
        labels[e,label[e]]=1
    return labels

# In[]
if __name__ == "__main__":
    xs=read_data(r"F:\研究生书籍\研究生书籍\深度学习-tensorflow\transfer_learning_vgg16\image1000test200\train")
    ys=read_label(r'F:\研究生书籍\研究生书籍\深度学习-tensorflow\transfer_learning_vgg16\image1000test200\train.txt')
    xs_val=read_data(r"F:\研究生书籍\研究生书籍\深度学习-tensorflow\transfer_learning_vgg16\image1000test200\test")
    ys_val=read_label(r'F:\研究生书籍\研究生书籍\深度学习-tensorflow\transfer_learning_vgg16\image1000test200\test.txt')    
    
    #pdb.set_trace()
    vgg = Vgg16(vgg16_npy_path=r'F:\研究生书籍\研究生书籍\深度学习-tensorflow\transfer_learning_vgg16\vgg16.npy')
    print('Net built')
    loss_curve=[]
    for i in range(100):
        b_idx = np.random.randint(0, len(xs), 16)
        train_loss = vgg.train(xs[b_idx], ys[b_idx])
        loss_curve.append(train_loss)
        print(i+1, 'train loss: ', train_loss)
        if (i) % 5 == 0:#输入样本太多就会爆内存 这里就输入20个进去
            c_idx = np.random.randint(0, len(xs_val), 20)
            y_op=vgg.predict_label(xs_val[c_idx])
            
            predicted_labels_val = np.argmax(y_op,1)
            labels = np.argmax(ys_val[c_idx],1)
            acc=np.sum(labels==predicted_labels_val)/len(labels)
            print('第',i+1,'次训练后的验证集分类精度为',acc)
            vgg.save('transfer_learn/')      # save learned fc layers

    predicted_label=[]
    for j in range(200):
        pre=vgg.predict_label(np.reshape(xs_val[j],[1,224,224,3]))
        predicted_label.append(np.argmax(pre,1))
    predicted_label_val=predicted_label
    label= np.argmax(ys_val,1)
    all_acc=np.sum(label==predicted_label_val)/len(label)
    print('结束训练后的验证集最终分类精度为',all_acc)
    vgg.save('transfer_learn_finall/')      # save learned fc layers
    plt.figure()
    plt.plot(loss_curve)
    plt.show()
    
    
    
    



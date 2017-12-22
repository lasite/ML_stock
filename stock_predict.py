import tensorflow as tf
import numpy as np
import pandas as pd
import math
import data_collecter_sohu as sohu_dc
pd.set_option('display.width',180)  
df_002024=sohu_dc.get_hist_data('002024','20080101','20171215')
df_002024_norm=(df_002024-df_002024.min())/(df_002024.max()-df_002024.min())
data=df_002024_norm['20171215':'20161124']
features=[]
labels=[]
date=[]
for index in range(data.index.size):
    slice=data[data.index.size-6-index:data.index.size-index]
    if(slice.index.size<6):
        break
    features.append(slice[1:6].values.flatten())
    date.append(slice.index[0])
    if (slice.close[0]-slice.close[1])/slice.close[1]>0.03:
        labels.append(0)
    else:
        labels.append(1)

features_placeholder = tf.placeholder(tf.float32, shape=(None,85))
                                                                            
with tf.name_scope('hidden1'):
    weights = tf.Variable(tf.truncated_normal([85, 70],stddev=1.0 / math.sqrt(float(85))),name='weights')
    biases = tf.Variable(tf.zeros([70]),name='biases')
    hidden1 = tf.nn.tanh(tf.matmul(features_placeholder, weights) + biases)
with tf.name_scope('hidden2'):
    weights = tf.Variable(
    tf.truncated_normal([70, 20],stddev=1.0 / math.sqrt(float(70))),name='weights')
    biases = tf.Variable(tf.zeros([20]),name='biases')
    hidden2 = tf.nn.tanh(tf.matmul(hidden1, weights) + biases)
with tf.name_scope('softmax_linear'):
    weights = tf.Variable(tf.truncated_normal([20, 2],stddev=1.0 / math.sqrt(float(20))),name='weights')
    biases = tf.Variable(tf.zeros([2]),name='biases')
    logits = tf.matmul(hidden2, weights) + biases
y = tf.nn.softmax(logits)
values, indices = tf.nn.top_k(y, 2)
table = tf.contrib.lookup.index_to_string_table_from_tensor(tf.constant(['up','down']))
prediction_classes = table.lookup(tf.to_int64(indices))

correct = tf.nn.in_top_k(logits, labels, 1)
correct_count=tf.reduce_sum(tf.cast(correct, tf.int32))

sess=tf.Session()
saver = tf.train.Saver(tf.global_variables())

saver.restore(sess,'/tmp/tensorflow/stock/logs/fully_connected_feed/model.ckpt-399999')


a,b,c=sess.run([y,correct,correct_count],{features_placeholder:features})
print(a,b,c)

'''a=sess.run(logits,{features_placeholder:features})
for i in range(a.shape[0]):
    y = np.exp(a[i])
    y1=y[0]/y.sum()
    y2=y[1]/y.sum()
    print(date[i],y1,y2)'''

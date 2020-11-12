# encoding=utf-8

"""
bert-idcnn-crf layer
@Author:Chekecheke
"""

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import crf

class IDCNN_CRF(object):
    def __init__(self, embedded_chars,droupout_rate,
                 initializers,num_labels, seq_length, labels, lengths, is_training):
        """
        IDCNN-CRF 网络
        :param embedded_chars: Fine-tuning embedding input
        :param droupout_rate: droupout rate
        :param initializers: variable init class
        :param num_labels: 标签数量
        :param seq_length: 序列最大长度
        :param labels: 真实标签
        :param lengths: [batch_size] 每个batch下序列的真实长度
        :param is_training: 是否是训练过程
        """
        
        self.droupout_rate = droupout_rate
        self.embedded_chars = embedded_chars
        self.initializers = initializers
        self.seq_length = seq_length
        self.num_labels = num_labels
        self.labels = labels
        self.lengths = lengths
        self.filter_width = 3
        self.num_filter = 128
        self.is_training = is_training
        self.repeat_times = 4
        self.layers = [
            {
                'dilation': 1
            },
            {
                'dilation': 1
            },
            {
                'dilation': 2
            },
        ]

    def add_idcnn_crf_layer(self):
        """
        idcnn-crf网络
        :return: 
        """
        if self.is_training:
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.droupout_rate)
        #idcnn
        idcnn_output = self.IDCNN_layer(self.embedded_chars)
        #project
        logits = self.project_layer_idcnn(idcnn_output)
        print("logits shape:",logits.get_shape())
        #crf
        loss, trans = self.crf_layer(logits)
        # CRF decode, pred_ids 是一条最大概率的标注路径
        pred_ids, _ = crf.crf_decode(potentials=logits, transition_params=trans, sequence_length=self.lengths)
        return ((loss, logits, trans, pred_ids))

    def bias_variable(self,shape_b):
        initial = tf.constant(0.1,shape=shape_b)
        return tf.Variable(initial)

    def w_variable(self,shape_w):
        W = tf.get_variable(
                            name = 'filter_W',
                            shape = shape_w,
                            initializer = tf.contrib.layers.xavier_initializer())
        return W

    def identity_block(self,x_input,dilation,shape_w,shape_b,stage,block):
        block_name = 'res' + str(stage) + block
        with tf.variable_scope(block_name,reuse =tf.AUTO_REUSE):
            x_shortcut = x_input
            x_shape = x_input.get_shape()
            params_shape = x_shape[-1]

            beta = tf.get_variable('beta',params_shape,initializer=tf.constant_initializer(0))
            gamma = tf.get_variable('gama',params_shape,initializer=tf.constant_initializer(1))

            # moving_mean = tf.get_variable('moving_mean',params_shape,initializer=tf.zeros_initializer(),trainable=False)
            # moving_var = tf.get_variable('moving_var',params_shape,initializer=tf.ones_initializer(),trainable=False)

            axes = list(range(len(x_shape)-1))
            batch_mean,batch_var = tf.nn.moments(x_input,axes,name='moment')
            ema = tf.train.ExponentialMovingAverage(0.9)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean,batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean),tf.identity(batch_var)

            mean,var = tf.cond(tf.equal(True,True),mean_var_with_update,lambda:(ema.average(batch_mean),ema.average(batch_var)))



            x = tf.nn.atrous_conv2d(
                                     value = x_input,
                                     filters = self.w_variable(shape_w),
                                     rate = dilation[0],
                                     padding = 'SAME')
            x = tf.nn.atrous_conv2d(
                                    value = x,
                                    filters = self.w_variable(shape_w),
                                    rate = dilation[1],
                                    padding = 'SAME')
            x = tf.nn.atrous_conv2d(
                                    value = x,
                                    filters = self.w_variable(shape_w),
                                    rate = dilation[2],
                                    padding = 'SAME')
            x = tf.nn.batch_normalization(
                                           x = x,
                                           mean = mean,
                                           variance = var,
                                           offset = beta,
                                           scale = gamma,
                                           variance_epsilon = 1e-5)
            add = tf.add(x,x_shortcut)
            b_conv_fin = self.bias_variable(shape_b)
            add_result = tf.nn.leaky_relu(add+b_conv_fin)
        return add_result


    def IDCNN_layer(self, model_inputs,
                    name=None):
        """
        :param idcnn_inputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, cnn_output_width]
        """
        model_inputs = tf.expand_dims(model_inputs, 1)
        self.embedding_dim = model_inputs.get_shape()[-1]
        reuse = False
        with tf.variable_scope("idcnn" if not name else name):
            shape = [1, self.filter_width, self.embedding_dim,
                     self.num_filter]
            print(shape)
            filter_weights = tf.get_variable(
                "idcnn_filter",
                shape=[1, self.filter_width, self.embedding_dim,
                       self.num_filter],
                initializer=self.initializers.xavier_initializer())

            """
            shape of input = [batch, in_height, in_width, in_channels]
            shape of filter = [filter_height, filter_width, in_channels, out_channels]
            """
            layerInput = tf.nn.conv2d(model_inputs,
                                      filter_weights,
                                      strides=[1, 1, 1, 1],
                                      padding="SAME",
                                      name="init_layer")
            finalOutFromLayers = []
            totalWidthForLastDim = 0
            for j in range(self.repeat_times):
                for i in range(len(self.layers)):
                    dilation = self.layers[i]['dilation']
                    isLast = True if i == (len(self.layers) - 1) else False
                    with tf.variable_scope("atrous-conv-layer-%d" % i,
                                           reuse=tf.AUTO_REUSE):
                        w = tf.get_variable(
                            "filterW",
                            shape=[1, self.filter_width, self.num_filter,
                                   self.num_filter],
                            initializer=tf.contrib.layers.xavier_initializer())
                        b = tf.get_variable("filterB", shape=[self.num_filter])
                        conv = tf.nn.atrous_conv2d(layerInput,
                                                   w,
                                                   rate=dilation,
                                                   padding="SAME")
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv)
                        if isLast:
                            finalOutFromLayers.append(conv)
                            totalWidthForLastDim += self.num_filter
                        layerInput = conv
            finalOut = tf.concat(axis=3, values=finalOutFromLayers)
            keepProb = 1.0 if reuse else 0.5
            finalOut = tf.nn.dropout(finalOut, keepProb)

            finalOut = tf.squeeze(finalOut, [1])
            finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])
            self.cnn_output_width = totalWidthForLastDim
            return finalOut
        
    def project_layer_idcnn(self, idcnn_outputs, name=None):
        """
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.cnn_output_width, self.num_labels],
                                    dtype=tf.float32, initializer=self.initializers.xavier_initializer())

                b = tf.get_variable("b",  initializer=tf.constant(0.001, shape=[self.num_labels]))

                pred = tf.nn.xw_plus_b(idcnn_outputs, W, b)

            return tf.reshape(pred, [-1,self.seq_length, self.num_labels])

    def crf_layer(self, logits):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable(
                "transitions",
                shape=[self.num_labels, self.num_labels],
                initializer=self.initializers.xavier_initializer())
            log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                inputs=logits,
                tag_indices=self.labels,
                transition_params=trans,
                sequence_lengths=self.lengths)
            return tf.reduce_mean(-log_likelihood), trans

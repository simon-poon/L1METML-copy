from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, BatchNormalization, Dropout, Lambda, Conv1D, SpatialDropout1D, Concatenate, Flatten, Reshape, Multiply, Add, GlobalAveragePooling1D, Activation, Permute, Softmax, Layer
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow import slice
from tensorflow.keras import initializers
import qkeras
from qkeras.qlayers import QDense, QActivation
import numpy as np
import itertools

class weighted_sum_layer(Layer):
    '''Either does weight times inputs
    or weight times inputs + bias
    Input to be provided as:
      - Weights
      - ndim biases (if applicable)
      - ndim items to sum
    Currently works for 3-dim input, summing over the 2nd axis'''
    #def __init__(self, ndim=2, with_bias=False, **kwargs):
    #    super(weighted_sum_layer, self).__init__(**kwargs)
    #    self.with_bias = with_bias
    #    self.ndim = ndim
#
    #def get_config(self):
    #    cfg = super(weighted_sum_layer, self).get_config()
    #    cfg['ndim'] = self.ndim
    #    cfg['with_bias'] = self.with_bias
    #    return cfg

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)

class Custom_Multiply(Layer):
    #def __init__(self, num_of_bins, **kwargs):
    #    super(Custom_Multiply, self).__init__(**kwargs)
    #    self.num_of_bins = num_of_bins

    def call(self, weights, pxpy): #(Batch x 100 x num_of_bins)
        px = tf.gather(pxpy, [0], axis=-1)
        py = tf.gather(pxpy, [1], axis=-1)
        weighted_px = tf.multiply(weights,px)
        weighted_py = tf.multiply(weights,py)
        weighted_pxpy = tf.stack([weighted_px,weighted_py],axis=-2)
        return weighted_pxpy

    #def get_config(self):
    #    cfg = super(Custom_Multiply, self).get_config()
    #    cfg['num_of_bins'] = self.num_of_bins
    #    return cfg

class add_axis(Layer):
    
    def call(self, inputs):
        exp_dim = tf.expand_dims(inputs, axis=-1)
        #repeat = tf.repeat(exp_dim, 2, axis=-1)
        #reshape = tf.reshape(repeat, [None, 100, 2, num_of_bins])
        return exp_dim

class Bin_Multiply(Layer):
    def __init__(self, num_of_bins=50, **kwargs):
        super(Bin_Multiply, self).__init__(**kwargs)
        self.num_of_bins = num_of_bins
        bins = np.linspace(-200,200, self.num_of_bins-1)
        bin_center = bins - (bins[1] - bins[0])/2
        self.bin_center = np.append(bin_center, 200 + (bins[1] - bins[0])/2)
        #self.bin_center = tf.convert_to_tensor(bin_center, dtype=tf.float32)
    
    def call(self, prob):
        #px = tf.gather(self.bin_center, [0], axis=-1)
        #py = tf.gather(self.bin_center, [1], axis=-1)
        bin_center = tf.convert_to_tensor(self.bin_center, dtype=tf.float32)

        x_list = list(range(0,self.num_of_bins*2,2))
        y_list = list(range(1,self.num_of_bins*2,2))

        prob_x = tf.gather(prob, [0], axis=1)
        #prob_x = tf.squeeze(prob_x, axis=-2)
        prob_y = tf.gather(prob, [1], axis=1)


        pred_px = bin_center * prob_x
        pred_py = bin_center * prob_y
        pred_px = tf.reduce_sum(pred_px, axis=-1)
        pred_py = tf.reduce_sum(pred_py, axis=-1)
        pred_pxpy = tf.concat([pred_px, pred_py], axis=-1)
        return pred_pxpy

    def get_config(self):
        cfg = super(Bin_Multiply, self).get_config()
        cfg['num_of_bins'] = self.num_of_bins
        cfg['bin_center'] = self.bin_center
        return cfg

def dense_embedding(n_features=6,
                    n_features_cat=2,
                    activation='relu',
                    number_of_pupcandis=100,
                    embedding_input_dim={0: 13, 1: 3},
                    emb_out_dim=8,
                    with_bias=True,
                    t_mode=0,
                    units=[64, 32, 16],
                    num_of_bins=50):
    n_dense_layers = len(units)

    inputs_cont = Input(shape=(number_of_pupcandis, n_features-2), name='input_cont')
    pxpy = Input(shape=(number_of_pupcandis, 2), name='input_pxpy')

    embeddings = []
    inputs = [inputs_cont, pxpy]
    for i_emb in range(n_features_cat):
        input_cat = Input(shape=(number_of_pupcandis, ), name='input_cat{}'.format(i_emb))
        inputs.append(input_cat)
        embedding = Embedding(
            input_dim=embedding_input_dim[i_emb],
            output_dim=emb_out_dim,
            embeddings_initializer=initializers.RandomNormal(
                mean=0,
                stddev=0.4/emb_out_dim),
            name='embedding{}'.format(i_emb))(input_cat)
        embeddings.append(embedding)

    # can concatenate all 3 if updated in hls4ml, for now; do it pairwise
    # x = Concatenate()([inputs_cont] + embeddings)
    emb_concat = Concatenate()(embeddings)
    x = Concatenate()([inputs_cont, emb_concat])

    for i_dense in range(n_dense_layers):
        x = Dense(units[i_dense], activation='linear', kernel_initializer='lecun_uniform')(x)
        x = BatchNormalization(momentum=0.95)(x)
        x = Activation(activation=activation)(x)

    if t_mode == 0:
        x = GlobalAveragePooling1D(name='pool')(x)
        x = Dense(2, name='output', activation='linear')(x)

    if t_mode == 1:
        if with_bias:
            b = Dense(2, name='met_bias', activation='linear', kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
            pxpy = Add()([pxpy, b])
        w = Dense(num_of_bins, name='met_weight', activation='linear', kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
        w = BatchNormalization(trainable=False, name='met_weight_minus_one', epsilon=False)(w)
        x = Custom_Multiply()(w, pxpy)
        x = weighted_sum_layer(name='weighted_sum_layer')(x)
        w = Softmax(axis=-1)(x)
        m = Bin_Multiply(num_of_bins=num_of_bins)(w)

        #m = add_axis()(x)

        #units_list = [32,32]
        #for i_dense in range(2):
        #    m = Dense(units_list[i_dense], activation='linear', kernel_initializer='lecun_uniform')(m)
        #    m = BatchNormalization(momentum=0.95)(m)
        #    m = Activation(activation=activation)(m)
        #m = Dense(num_of_bins, activation='linear', kernel_initializer='lecun_uniform')(m)
        #w = BatchNormalization(momentum=0.95)(m)
        #w = Softmax(axis=-1)(w)
        #m = Bin_Multiply(num_of_bins=num_of_bins)(w)


    outputs = [m,w]

    keras_model = Model(inputs=inputs, outputs=outputs)
    a = np.ones(num_of_bins)
    b = -a
    c = np.zeros(num_of_bins)
    d = a
    keras_model.get_layer('met_weight_minus_one').set_weights([a, b, c, d])

    return keras_model


def dense_embedding_quantized(n_features=6,
                              n_features_cat=2,
                              number_of_pupcandis=100,
                              embedding_input_dim={0: 13, 1: 3},
                              emb_out_dim=2,
                              with_bias=True,
                              t_mode=0,
                              logit_total_bits=7,
                              logit_int_bits=2,
                              activation_total_bits=7,
                              logit_quantizer='quantized_bits',
                              activation_quantizer='quantized_relu',
                              activation_int_bits=2,
                              alpha=1,
                              use_stochastic_rounding=False,
                              units=[64, 32, 16]):
    n_dense_layers = len(units)

    logit_quantizer = getattr(qkeras.quantizers, logit_quantizer)(logit_total_bits, logit_int_bits, alpha=alpha, use_stochastic_rounding=use_stochastic_rounding)
    activation_quantizer = getattr(qkeras.quantizers, activation_quantizer)(activation_total_bits, activation_int_bits)

    inputs_cont = Input(shape=(number_of_pupcandis, n_features-2), name='input_cont')
    pxpy = Input(shape=(number_of_pupcandis, 2), name='input_pxpy')

    embeddings = []
    inputs = [inputs_cont, pxpy]
    for i_emb in range(n_features_cat):
        input_cat = Input(shape=(number_of_pupcandis, ), name='input_cat{}'.format(i_emb))
        inputs.append(input_cat)
        embedding = Embedding(
            input_dim=embedding_input_dim[i_emb],
            output_dim=emb_out_dim,
            embeddings_initializer=initializers.RandomNormal(
                mean=0,
                stddev=0.4/emb_out_dim),
            name='embedding{}'.format(i_emb))(input_cat)
        embeddings.append(embedding)

    # can concatenate all 3 if updated in hls4ml, for now; do it pairwise
    # x = Concatenate()([inputs_cont] + embeddings)
    emb_concat = Concatenate()(embeddings)
    x = Concatenate()([inputs_cont, emb_concat])

    for i_dense in range(n_dense_layers):
        x = QDense(units[i_dense], kernel_quantizer=logit_quantizer, bias_quantizer=logit_quantizer, kernel_initializer='lecun_uniform')(x)
        x = BatchNormalization(momentum=0.95)(x)
        x = QActivation(activation=activation_quantizer)(x)

    if t_mode == 0:
        x = qkeras.qpooling.QGlobalAveragePooling1D(name='pool', quantizer=logit_quantizer)(x)
        # pool size?
        outputs = QDense(2, name='output', bias_quantizer=logit_quantizer, kernel_quantizer=logit_quantizer, activation='linear')(x)

    if t_mode == 1:
        if with_bias:
            b = QDense(2, name='met_bias', kernel_quantizer=logit_quantizer, bias_quantizer=logit_quantizer, kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
            pxpy = Add()([pxpy, b])
        w = QDense(1, name='met_weight', kernel_quantizer=logit_quantizer, bias_quantizer=logit_quantizer, kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
        w = BatchNormalization(trainable=False, name='met_weight_minus_one', epsilon=False)(w)
        x = Multiply()([w, pxpy])

        x = GlobalAveragePooling1D(name='output')(x)
    outputs = x

    keras_model = Model(inputs=inputs, outputs=outputs)

    keras_model.get_layer('met_weight_minus_one').set_weights([np.array([1.]), np.array([-1.]), np.array([0.]), np.array([1.])])

    return keras_model


# Create the sender and receiver relations matrices
def assign_matrices(N, Nr):
    Rr = np.zeros([N, Nr], dtype=np.float32)
    Rs = np.zeros([N, Nr], dtype=np.float32)
    receiver_sender_list = [i for i in itertools.product(range(N), range(N)) if i[0] != i[1]]
    for i, (r, s) in enumerate(receiver_sender_list):
        Rr[r, i] = 1
        Rs[s, i] = 1
    return Rs, Rr

import tensorflow as tf
from tensorflow.keras.layers import Layer


def graph_embedding(compute_ef, n_features=6,
                    n_features_cat=2,
                    activation='relu',
                    number_of_pupcandis=100,
                    embedding_input_dim={0: 13, 1: 3},
                    emb_out_dim=8,
                    units=[64, 32, 16],
                    edge_list=[]):
    n_dense_layers = len(units)
    name = 'met'

    inputs_cont = Input(shape=(number_of_pupcandis, n_features-2), name='input_cont')
    pxpy = Input(shape=(number_of_pupcandis, 2), name='input_pxpy')

    embeddings = []
    inputs = [inputs_cont, pxpy]
    for i_emb in range(n_features_cat):
        input_cat = Input(shape=(number_of_pupcandis, ), name='input_cat{}'.format(i_emb))
        inputs.append(input_cat)
        embedding = Embedding(
            input_dim=embedding_input_dim[i_emb],
            output_dim=emb_out_dim,
            embeddings_initializer=initializers.RandomNormal(
                mean=0,
                stddev=0.4/emb_out_dim),
            name='embedding{}'.format(i_emb))(input_cat)
        embeddings.append(embedding)

    N = number_of_pupcandis
    Nr = N*(N-1)
    if compute_ef == 1:
        num_of_edge_feat = len(edge_list)
        edge_feat = Input(shape=(Nr, num_of_edge_feat), name='edge_feat')
        inputs.append(edge_feat)

    # can concatenate all 3 if updated in hls4ml, for now; do it pairwise
    # x = Concatenate()([inputs_cont] + embeddings)
    emb_concat = Concatenate()(embeddings)
    x = Concatenate()([inputs_cont, emb_concat])

    N = number_of_pupcandis
    P = n_features+n_features_cat
    Nr = N*(N-1)  # number of relations (edges)

    x = BatchNormalization()(x)

    # Swap axes of input data (batch,nodes,features) -> (batch,features,nodes)
    x = Permute((2, 1), input_shape=x.shape[1:])(x)

    # Marshaling function
    ORr = Dense(Nr, use_bias=False, trainable=False, name='tmul_{}_1'.format(name))(x)  # Receiving adjacency matrix
    ORs = Dense(Nr, use_bias=False, trainable=False, name='tmul_{}_2'.format(name))(x)  # Sending adjacency matrix
    node_feat = Concatenate(axis=1)([ORr, ORs])  # Concatenates Or and Os  ( no relations features Ra matrix )
    # Outputis new array = [batch, 2x features, edges]

    # Edges MLP
    h = Permute((2, 1), input_shape=node_feat.shape[1:])(node_feat)
    edge_units = [64, 32, 16]
    n_edge_dense_layers = len(edge_units)
    if compute_ef == 1:
        h = Concatenate(axis=2, name='concatenate_edge')([h, edge_feat])
    for i_dense in range(n_edge_dense_layers):
        h = Dense(edge_units[i_dense], activation='linear', kernel_initializer='lecun_uniform')(h)
        h = BatchNormalization(momentum=0.95)(h)
        h = Activation(activation=activation)(h)
    out_e = h

    # Transpose output and permutes columns 1&2
    out_e = Permute((2, 1))(out_e)

    # Multiply edges MLP output by receiver nodes matrix Rr
    out_e = Dense(N, use_bias=False, trainable=False, name='tmul_{}_3'.format(name))(out_e)

    # Nodes MLP (takes as inputs node features and embeding from edges MLP)
    inp_n = Concatenate(axis=1)([x, out_e])

    # Transpose input and permutes columns 1&2
    h = Permute((2, 1), input_shape=inp_n.shape[1:])(inp_n)

    # Nodes MLP
    for i_dense in range(n_dense_layers):
        h = Dense(units[i_dense], activation='linear', kernel_initializer='lecun_uniform')(h)
        h = BatchNormalization(momentum=0.95)(h)
        h = Activation(activation=activation)(h)
    w = Dense(1, name='met_weight', activation='linear', kernel_initializer=initializers.VarianceScaling(scale=0.02))(h)
    w = BatchNormalization(trainable=False, name='met_weight_minus_one', epsilon=False)(w)
    x = Multiply()([w, pxpy])
    outputs = weighted_sum_layer(name='output')(x)
    #outputs = GlobalAveragePooling1D(name='output')(x)

    keras_model = Model(inputs=inputs, outputs=outputs)

    keras_model.get_layer('met_weight_minus_one').set_weights([np.array([1.]), np.array([-1.]), np.array([0.]), np.array([1.])])

    # Create a fully connected adjacency matrix
    Rs, Rr = assign_matrices(N, Nr)
    keras_model.get_layer('tmul_{}_1'.format(name)).set_weights([Rr])
    keras_model.get_layer('tmul_{}_2'.format(name)).set_weights([Rs])
    keras_model.get_layer('tmul_{}_3'.format(name)).set_weights([np.transpose(Rr)])

    return keras_model

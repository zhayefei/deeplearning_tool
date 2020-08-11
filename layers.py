#! encoding=utf-8
from keras.layers.core import RepeatVector
from keras.engine.topology import Layer
from keras import backend as K
from keras.layers.core import Lambda
from keras.layers import Multiply, Lambda
import keras.initializers

my_multiply = Lambda(lambda x: Multiply()([x[0], x[1]]), name=u'matrix_multiply')
trilinear_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1), name=u'trilinear_concat')


class TrilinearAttentionLayer(Layer):
    
    def __init__(self, output_dim, time_step, pmpt_input_dim=0, rspn_input_dim=0, **kwargs):
        self.output_dim = output_dim
        self.time_step = time_step
        if pmpt_input_dim:
            self.pmpt_input_dim = pmpt_input_dim
        else:
            self.pmpt_input_dim = output_dim
        if rspn_input_dim:
            self.rspn_input_dim = rspn_input_dim
        else:
            self.rspn_input_dim = output_dim
        self.init = u'he_uniform'
        super(TrilinearAttentionLayer, self).__init__(**kwargs)

    def get_config(self):
        config = {u'output_dim': self.output_dim,
                  u'time_step': self.time_step,
                  u'pmpt_input_dim': self.pmpt_input_dim,
                  u'rspn_input_dim': self.rspn_input_dim}
        base_config = super(TrilinearAttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        print('(Attn build) Attention layer input shape: ', input_shape)  # [(None, 120, 100), (None, 100)]
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name=u'weight_tri',
                                    shape=(self.rspn_input_dim * 3, 1),
                                    initializer=self.init,
                                    trainable=True)

        print('(Attn build) Weight shape W_tri:', self.W.shape)  # (1, 100, 100)

        self.built = True

    def call(self, inputs, **kwargs):
        response_input = inputs[0]
        prompt_input = inputs[1]

        repeated_prompt_input = RepeatVector(self.time_step)(prompt_input)
        print('(Attn call) Repeated prompt input shape: ', repeated_prompt_input.shape)  # (?, 120, 100)
        print('(Attn call) Response input shape: ', response_input.shape)  # (?, ?, 100)

        prompt_response_pair = my_multiply([repeated_prompt_input, response_input])
        print('(Attn call) prompt_response_pair shape', prompt_response_pair.shape)  # (?, 120, 100)
        tri_concat= trilinear_concat([trilinear_concat([repeated_prompt_input, response_input]), prompt_response_pair])
        
        states = K.dot(tri_concat, self.W)   #[?, 280, 200] [?, 200, 100]
        print('(Attn call) states shape', states.shape)  # (?, 120, 1)

        alpha = K.exp(states)
        print('(Attn call) alpha shape', alpha.shape)  # (?, 120, 1)
        alpha_sum = K.sum(alpha, axis=1, keepdims=True) + K.epsilon()
        print('(Attn call) alpha sum shape', alpha_sum.shape)  # (?, 1, 1)
        alpha = alpha / alpha_sum
        print('(Attn call) alpha shape divided', alpha.shape)  # (?, 30, 50, 1)

        print('(Attn call) alpha shape', alpha.shape)  # (?, 120, 1)

        return alpha

class GatedLayer(Layer):

    def __init__(self, output_dim=0, **kwargs):
        self.output_dim = output_dim
        self.init = u'he_uniform'
        super(GatedLayer, self).__init__(**kwargs)

    def get_config(self):
        config = {u'output_dim': self.output_dim}
        base_config = super(GatedLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.W_g = self.add_weight(name=u'weight_gate',
                                    shape=(self.output_dim, self.output_dim),
                                    initializer=u'zeros',
                                    trainable=True)
        
        self.B = self.add_weight(name=u'bias',
                                shape=(1, self.output_dim),
                                initializer=u'zeros',
                                trainable=True)
 
        print('(GateLayer build) Weight shape W_g:', self.W_g.shape, self.B)
        self.built = True

    def call(self, input_vec, **kwargs):

        print('(GateLayer call) Wg shape', self.W_g.shape)  # 
        Wg_state = K.dot(input_vec, self.W_g)
        print('(GateLayer call) Wg_state shape', Wg_state.shape)  #
        gate_state = K.sigmoid(Wg_state + self.B)
        print('(GateLayer call) gate_state shape', gate_state.shape)  # 
        gated_input_vec = my_multiply([gate_state, input_vec])
        print('(GateLayer call) gated_input_vec shape', gated_input_vec.shape)  #
        return gated_input_vec


    def compute_mask(self, x, mask=None):
        return None

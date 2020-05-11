from keras.models import load_model,Model
from keras.layers import Conv2D, concatenate, Dense, Flatten,Input,Activation,Lambda

from keras.utils.vis_utils import plot_model
import keras.backend as K
import numpy as np


class icm_model():
    def __init__(self, input_shape, n_actions):
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.fm1 = self.feature_model()
        # self.fm2 = self.feature_model()
        # self.f_model = self.forward_model()
        # self.i_model = self.inverse_model()
        self.icm_model = self.create_icm_model()
        # self.softmaxize = self.softmax_layer()

    def feature_model(self):
        input1 = Input(self.input_shape)
        x = Conv2D(8, kernel_size=3, activation='relu', data_format='channels_first')(input1)
        x = Conv2D(16, kernel_size=2, activation='relu', data_format='channels_first')(x)
        y = Flatten()(x)

        model = Model(inputs=input1, outputs=y)
        # plot_model(model,'text.png',show_shapes=True)
        return model



    def create_inverse_model(self):

        def func(ft0, ft1):
            x = concatenate([ft0, ft1])
            x = Dense(32, activation='relu')(x)
            x = Dense(16, activation='relu')(x)
            h = Dense(self.n_actions, activation='softmax')(x)
            return h

        return func

    def create_forward_model(self):
        def func(obs1,act_input):
            print(obs1)
            x = concatenate([obs1, act_input])

            x = Dense(64, activation='relu')(x)
            x = Dense(32, activation='relu')(x)

            y = Dense(2304, activation='relu')(x)
            return y
        return func

    def create_softmax(self):
        def func(input):
            x = Activation('softmax')(input)
            return x
        return func

    def get_intrinsic_reward(self,x):
        return K.function([self.icm_model.get_layer("state_t").input,
                       self.icm_model.get_layer("state_t_1").input,
                       self.icm_model.get_layer("action_t").input],
                      [self.icm_model.get_layer("intrinsic_reward").output])(x)[0]

    def create_icm_model(self,beta=0.2,lam=1):

        s_t0 = Input(shape=self.input_shape, name='state_t')
        s_t1 = Input(shape=self.input_shape, name='state_t_1')
        act_t = Input(shape=(self.n_actions,),name='action_t')

        fmap = self.fm1
        f_t0 = fmap(inputs=s_t0)
        f_t1 = fmap(inputs=s_t1)

        act_predict = self.create_inverse_model()(f_t0, f_t1)

        f_t1_predict = self.create_forward_model()(f_t0,act_t)

        act_actual = self.create_softmax()(act_t)
        # add K.epsilon for avoiding log 0 case ,if it happens
        inverse_loss = Lambda(lambda x: -K.sum(x[0] * K.log(x[1] + K.epsilon()), axis=-1), output_shape=(1,))
        forward_loss = Lambda(lambda x: 0.5 * K.sum(K.square(x[0] - x[1]), axis=-1),output_shape=(1,))
        l2_loss = forward_loss([f_t1,f_t1_predict])
        l1_loss = inverse_loss([act_actual,act_predict])
        total_loss = Lambda(lambda x: beta * x[0] + (1.0 - beta) * x[1], output_shape=(1,), name='intrinsic_reward')
        l_ = total_loss([l2_loss,l1_loss])

        r_e = Input(shape=(1,))
        global_loss = Lambda(lambda x: (-lam * x[0] + x[1]), output_shape=(1,))
        final_loss = global_loss([r_e, l_])

        return Model([s_t0, s_t1, act_t, r_e], final_loss)



if __name__ == '__main__':
    input_shape = (3,15,15)
    n_actions = 5

    obs1 = np.random.rand(3, 15, 15)
    obs2 = np.random.rand(3, 15, 15)
    icm_action = np.zeros(n_actions)
    icm_action[1] = 1

    a = icm_action.reshape(1, -1)
    print(a,icm_action)




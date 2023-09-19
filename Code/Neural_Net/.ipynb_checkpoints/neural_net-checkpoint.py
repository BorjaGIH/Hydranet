from shared_pkgs.imports import *

# Define epsilon layer
# Define epsilon layer
class EpsilonLayer(Layer):

    def __init__(self):
        super(EpsilonLayer, self).__init__()

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.epsilon = self.add_weight(name='epsilon',
                                       shape=[1, 4],
                                       #initializer='RandomNormal',
                                       initializer='zeros',
                                       trainable=True)
        super(EpsilonLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        # import ipdb; ipdb.set_trace()
        return self.epsilon * tf.ones_like(inputs)[:, 0:4]
    
# Define Hydranet architecture
def make_hydranet(input_dim, num_treats, reg_l2):
    """
    Neural net predictive model. The dragon has many heads.
    :param input_dim:
    :param reg:
    :return:
    """
    inputs = Input(shape=(input_dim,), name='input')

    # representation
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(inputs)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
    
    # G propensity score (Rt vector)
    t_predictions = Dense(units=num_treats, activation = 'softmax')(x)

    # HYPOTHESIS
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)
    y2_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)
    y3_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)
    y4_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)

    # second layer
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)
    y2_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y2_hidden)
    y3_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y3_hidden)
    y4_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y4_hidden)

    # third layer
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(y0_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(y1_hidden)
    y2_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y2_predictions')(y2_hidden)
    y3_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y3_predictions')(y3_hidden)
    y4_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y4_predictions')(y4_hidden)

    # epsilons
    dl = EpsilonLayer()
    epsilons = dl(t_predictions, name='epsilon')

    if num_treats==5:
        concat_pred = Concatenate(1)([y0_predictions, y1_predictions, y2_predictions, y3_predictions, y4_predictions, t_predictions, epsilons])
    elif num_treats==10:
        concat_pred = Concatenate(1)([y0_predictions, y1_predictions, y2_predictions, y3_predictions, y4_predictions,\
                                      y5_predictions, y6_predictions, y7_predictions, y8_predictions, y9_predictions, t_predictions, epsilons])
    model = Model(inputs=inputs, outputs=concat_pred)

    return model

class EpsilonLayer_dr(Layer):

    def __init__(self):
        super(EpsilonLayer_dr, self).__init__()

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.epsilon = self.add_weight(name='epsilon',
                                       shape=[1, 1],
                                       initializer='RandomNormal',
                                       #  initializer='ones',
                                       trainable=True)
        super(EpsilonLayer_dr, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        # import ipdb; ipdb.set_trace()
        return self.epsilon * tf.ones_like(inputs)[:, 0:1]

def make_dragonnet(input_dim, reg_l2):

    inputs = Input(shape=(input_dim,), name='input')

    # representation
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(inputs)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)

    t_predictions = Dense(units=1, activation='sigmoid')(x)

    # HYPOTHESIS
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)

    # second layer
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)

    # third
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(
        y0_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(
        y1_hidden)

    dl = EpsilonLayer_dr()
    epsilons = dl(t_predictions, name='epsilon')
    # logging.info(epsilons)
    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons])
    model = Model(inputs=inputs, outputs=concat_pred)

    return model
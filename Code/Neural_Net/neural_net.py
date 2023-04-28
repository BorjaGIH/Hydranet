from shared_pkgs.imports import *

# Define epsilon layer
class EpsilonLayer(Layer): #careful, we modify this function
#class EpsilonLayer(Layer):

    def __init__(self):
        super(EpsilonLayer, self).__init__()

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.epsilon = self.add_weight(name='epsilon',
                                       shape=[1, 2],
                                       initializer='RandomNormal',
                                       #  initializer='ones',
                                       trainable=True)
        super(EpsilonLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        # import ipdb; ipdb.set_trace()
        return self.epsilon * tf.ones_like(inputs)[:, 0:2]
    
    
# Define Hydranet architecture
def make_dragonnet(input_dim, reg_l2):
    """
    Neural net predictive model. The dragon has many heads.
    :param input_dim:
    :param reg:
    :return:
    """
    t_l1 = 0.
    t_l2 = reg_l2
    inputs = Input(shape=(input_dim,), name='input')

    # representation
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(inputs)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
    
    # G propensity score (R3 vector)
    t_predictions = Dense(units=3, activation = 'softmax')(x) # correct is: softmax

    # HYPOTHESIS
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)
    y2_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)

    # second layer
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)
    y2_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(y2_hidden)

    # third layer
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(y0_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(y1_hidden)
    y2_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y2_predictions')(y2_hidden)

    
    # WARNING! Model misspecification activated
    #garb = inputs*tf.random.uniform([1,])
    #garbage_t = Dense(units=3, activation = 'softmax')(garb)
    
    
    # epsilons
    dl = EpsilonLayer()
    epsilons = dl(t_predictions, name='epsilon')
    #epsilons = dl(garbage_t, name='epsilon')

    
    #concat_pred = Concatenate(1)([y0_predictions, y1_predictions, y2_predictions, garbage_t, epsilons])
    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, y2_predictions, t_predictions, epsilons])
    model = Model(inputs=inputs, outputs=concat_pred)
    

    return model
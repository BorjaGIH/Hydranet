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

def build_tarnet(input_dim, output_dim, num_units=128, dropout=0.0, l2_weight=0.0, learning_rate=0.0001, num_layers=2,
                     num_treatments=2, p_ipm=0.5, imbalance_loss_weight=1.0, with_bn=False, with_propensity_dropout=True,
                     normalize=True,
                     **kwargs):

    rnaseq_input = Input(shape=(input_dim,))
    treatment_input = Input(shape=(1,), dtype="int32")
    model_outputs, loss_weights = [], []

    if with_propensity_dropout:
        dropout = 0
        propensity_output = ModelBuilder.build_mlp(rnaseq_input,
                                                   dim=num_units,
                                                   p_dropout=dropout,
                                                   num_layers=num_layers,
                                                   with_bn=with_bn,
                                                   l2_weight=l2_weight)
        propensity_output = Dense(num_treatments, activation="softmax", name="propensity")(propensity_output)
        model_outputs.append(propensity_output)
        loss_weights.append(1)
        gamma = 0.5
        propensity_dropout = Lambda(lambda x: tf.stop_gradient(x))(propensity_output)

        def get_treatment_propensities(x):
            cat_idx = tf.stack([tf.range(0, tf.shape(x[0])[0]), K.squeeze(tf.cast(x[1], "int32"), axis=-1)],
                               axis=1)
            return tf.gather_nd(x[0], cat_idx)

        propensity_dropout = Lambda(get_treatment_propensities)([propensity_dropout, treatment_input])
        propensity_dropout = Lambda(lambda x: 1. - gamma - 1./2. * (-x * tf.log(x) - (1 - x)*tf.log(1 - x)))\
            (propensity_dropout)
    else:
        propensity_dropout = None

    regulariser = None
    if imbalance_loss_weight != 0.0:

        def wasserstein_distance_regulariser(x):
            return imbalance_loss_weight*wasserstein(x, treatment_input, p_ipm,
                                                     num_treatments=num_treatments)

        regulariser = wasserstein_distance_regulariser

    # Build shared representation.
    last_layer = ModelBuilder.build_mlp(rnaseq_input,
                                        dim=num_units,
                                        p_dropout=dropout,
                                        num_layers=num_layers,
                                        with_bn=with_bn,
                                        l2_weight=l2_weight,
                                        propensity_dropout=propensity_dropout,
                                        normalize=normalize,
                                        last_activity_regulariser=regulariser)

    last_layer_h = last_layer

    all_indices, outputs = [], []
    for i in range(num_treatments):

        def get_indices_equal_to(x):
            return tf.reshape(tf.to_int32(tf.where(tf.equal(tf.reshape(x, (-1,)), i))), (-1,))

        indices = Lambda(get_indices_equal_to)(treatment_input)

        current_last_layer_h = Lambda(lambda x: tf.gather(x, indices))(last_layer_h)

        if with_propensity_dropout:
            current_propensity_dropout = Lambda(lambda x: tf.gather(propensity_dropout, indices))(propensity_dropout)
        else:
            current_propensity_dropout = None

        last_layer = ModelBuilder.build_mlp(current_last_layer_h,
                                            dim=num_units,
                                            p_dropout=dropout,
                                            num_layers=num_layers,
                                            with_bn=with_bn,
                                            propensity_dropout=current_propensity_dropout,
                                            l2_weight=l2_weight)

        output = Dense(output_dim, activation="linear", name="head_" + str(i))(last_layer)

        all_indices.append(indices)
        outputs.append(output)

    def do_dynamic_stitch(x):
        num_tensors = len(x)

        data_indices = map(tf.to_int32, x[:num_tensors/2])
        data = map(tf.to_float, x[num_tensors/2:])

        stitched = tf.dynamic_stitch(data_indices, data)
        return stitched

    output = Lambda(do_dynamic_stitch, name="dynamic_stitch")(all_indices + outputs)
    model_outputs.append(output)
    loss_weights.append(1)

    model = Model(inputs=[rnaseq_input, treatment_input],
                  outputs=model_outputs)
    model.summary()

    main_model = ModelBuilder.compile_model(model, learning_rate,
                                            loss_weights=loss_weights,
                                            main_loss="mse")

    return main_model
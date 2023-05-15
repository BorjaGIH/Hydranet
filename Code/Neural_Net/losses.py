from shared_pkgs.imports import *
############ HYDRANET LOSSES ##############

def regression_loss(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]
    y2_pred = concat_pred[:, 2]
    y3_pred = concat_pred[:, 3]
    y4_pred = concat_pred[:, 4]

    loss0 = tf.reduce_sum(tf.cast(t_true == 0, tf.float32) * tf.square(y_true - y0_pred))
    loss1 = tf.reduce_sum(tf.cast(t_true == 1, tf.float32) * tf.square(y_true - y1_pred))
    loss2 = tf.reduce_sum(tf.cast(t_true == 2, tf.float32) * tf.square(y_true - y2_pred))
    loss3 = tf.reduce_sum(tf.cast(t_true == 3, tf.float32) * tf.square(y_true - y3_pred))
    loss4 = tf.reduce_sum(tf.cast(t_true == 4, tf.float32) * tf.square(y_true - y4_pred))
    
    return loss0 + loss1 + loss2 + loss3 + loss4


def categorical_classification_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 5:10]

    temp = tf.one_hot(tf.cast(t_true, tf.int32), 5)
    temp = tf.cast(temp, tf.float32)

    temp2 = K.categorical_crossentropy(temp, t_pred)
    losst = tf.reduce_sum(temp2)

    return losst


def hydranet_loss(concat_true, concat_pred):
    return regression_loss(concat_true, concat_pred) + categorical_classification_loss(concat_true, concat_pred)


def treatment_accuracy(concat_true, concat_pred):  # For training monitoring purposes
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 5:10]

    temp = tf.one_hot(tf.cast(t_true, tf.int32), 5)
    temp = tf.cast(temp, tf.float32)

    return categorical_accuracy(temp, t_pred)


def track_epsilon(concat_true, concat_pred): # For training monitoring purposes
    epsilons = concat_pred[:, 10:14]
    return tf.abs(tf.reduce_mean(epsilons))


def make_tarreg_loss(ratio=1., hydranet_loss_=hydranet_loss): # Targeted regularization loss
    #print(hydranet_loss_)
    def tarreg_ATE_unbounded_domain_loss(concat_true, concat_pred):
        vanilla_loss = hydranet_loss_(concat_true, concat_pred)

        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        y2_pred = concat_pred[:, 2]
        y3_pred = concat_pred[:, 3]
        y4_pred = concat_pred[:, 4]
        t_pred = concat_pred[:, 5:10]

        epsilons = concat_pred[:, 10:14]
        #t_pred = (t_pred + 0.001) / 1.001
        # t_pred = tf.clip_by_value(t_pred,0.01, 0.99,name='t_pred')

        # 5-fold
        y_pred = tf.cast(t_true == 0, tf.float32) * y0_pred + tf.cast(t_true == 1, tf.float32) * y1_pred + tf.cast(t_true == 2, tf.float32) * y2_pred + tf.cast(t_true == 3, tf.float32) * y3_pred + tf.cast(t_true == 4,tf.float32) * y4_pred

        h = tf.transpose(
            [tf.cast(t_true == 1, tf.float32) / t_pred[:, 1] - tf.cast(t_true == 0, tf.float32) / t_pred[:, 0],
             tf.cast(t_true == 2, tf.float32) / t_pred[:, 2] - tf.cast(t_true == 0, tf.float32) / t_pred[:, 0],
             tf.cast(t_true == 3, tf.float32) / t_pred[:, 3] - tf.cast(t_true == 0, tf.float32) / t_pred[:, 0],
             tf.cast(t_true == 4, tf.float32) / t_pred[:, 4] - tf.cast(t_true == 0, tf.float32) / t_pred[:, 0]])


        y_pert = y_pred + tf.math.reduce_sum(tf.math.multiply(epsilons, h), axis=1)
        targeted_regularization = tf.reduce_sum(tf.square(y_true - y_pert))

        # Final
        loss = vanilla_loss + ratio * targeted_regularization
        return loss

    return tarreg_ATE_unbounded_domain_loss


############ DRAGONNET LOSSES ###############
def regression_loss_dr(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_pred))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_pred))

    return loss0 + loss1


def binary_classification_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    #t_pred = (t_pred + np.eps) / 1.001
    losst = tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))

    return losst


def dragonnet_loss_binarycross_dr(concat_true, concat_pred):
    return regression_loss_dr(concat_true, concat_pred) + binary_classification_loss(concat_true, concat_pred)


def treatment_accuracy_dr(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    #t_pred = (t_pred + 0.001) / 1.001
    return binary_accuracy(t_true, t_pred)


def track_epsilon_dr(concat_true, concat_pred):
    epsilons = concat_pred[:, 3]
    return tf.abs(tf.reduce_mean(epsilons))


def make_tarreg_loss_dr(ratio=1., dragonnet_loss=dragonnet_loss_binarycross_dr):
    def tarreg_ATE_unbounded_domain_loss_dr(concat_true, concat_pred):
        vanilla_loss = dragonnet_loss(concat_true, concat_pred)

        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]

        t_pred = concat_pred[:, 2]

        epsilons = concat_pred[:, 3]
        #t_pred = (t_pred + 0.01) / 1.01
        #t_pred = tf.clip_by_value(t_pred,0.01, 0.99,name='t_pred')

        y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

        h = t_true / t_pred - (1 - t_true) / (1 - t_pred)

        y_pert = y_pred + epsilons * h
        targeted_regularization = tf.reduce_sum(tf.square(y_true - y_pert))

        # final
        loss = vanilla_loss + ratio * targeted_regularization
        return loss

    return tarreg_ATE_unbounded_domain_loss_dr
from shared_pkgs.imports import *

def regression_loss(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]
    y2_pred = concat_pred[:, 2]
    
    loss0 = tf.reduce_sum(tf.cast(t_true==0, tf.float32) * tf.square(y_true - y0_pred))
    loss1 = tf.reduce_sum(tf.cast(t_true==1, tf.float32) * tf.square(y_true - y1_pred))
    loss2 = tf.reduce_sum(tf.cast(t_true==2, tf.float32) * tf.square(y_true - y2_pred))

    return loss0 + loss1 + loss2


def categorical_classification_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 3:6]
    t_pred = (t_pred + 0.001) / 1.002
    
    #temp = to_categorical(t_true, num_classes=3)
    #temp = to_categorical(t_true, num_classes=3, dtype='int')
    temp = tf.one_hot(tf.cast(t_true, tf.int32), 3)
    temp = tf.cast(temp, tf.float32)
    
    temp1 = tf.convert_to_tensor(temp)
    temp2 = K.categorical_crossentropy(temp1, t_pred)
    losst = tf.reduce_sum(temp2)

    return losst


def dragonnet_loss_cross(concat_true, concat_pred):
    return regression_loss(concat_true, concat_pred) + categorical_classification_loss(concat_true, concat_pred)



def binary_classification_loss(concat_true, concat_pred): # Deprecated
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 3]
    t_pred = (t_pred + 0.001) / 1.002
    losst = tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))

    return losst


def treatment_accuracy(concat_true, concat_pred): # For training monitoring purposes
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 3:6]
    
    temp = tf.one_hot(tf.cast(t_true, tf.int32), 3)
    temp = tf.cast(temp, tf.float32)

    return categorical_accuracy(temp, t_pred) # .numpy()


def track_epsilon(concat_true, concat_pred): # For training monitoring purposes
    epsilons = concat_pred[:, 6:8]
    #epsilons = concat_pred[:, 6]
    return tf.abs(tf.reduce_mean(epsilons))


def make_tarreg_loss(ratio=1., dragonnet_loss=dragonnet_loss_cross): # Targeted regularization loss
    def tarreg_ATE_unbounded_domain_loss(concat_true, concat_pred):
        vanilla_loss = dragonnet_loss(concat_true, concat_pred)

        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        y2_pred = concat_pred[:, 2]
        t_pred = concat_pred[:, 3:6]
        
        epsilons = concat_pred[:, 6:8]
        t_pred = (t_pred + 0.01) / 1.02
        # t_pred = tf.clip_by_value(t_pred,0.01, 0.99,name='t_pred')
        
        # Trinary
        y_pred = tf.cast(t_true==0, tf.float32) * y0_pred + tf.cast(t_true==1, tf.float32) * y1_pred + tf.cast(t_true==2, tf.float32) * y2_pred
        t_pred_current = tf.cast(t_true==0, tf.float32) * t_pred[:,0] + tf.cast(t_true==1, tf.float32) * t_pred[:,1] + tf.cast(t_true==2, tf.float32) * t_pred[:,2]
        
        # wrong_h =  1 # WARNING: Model misspecification activated
        h = tf.transpose([tf.cast(t_true==1, tf.float32)/t_pred[:,1] - tf.cast(t_true==0, tf.float32)/t_pred[:,0], tf.cast(t_true==2, tf.float32)/t_pred[:,2] - tf.cast(t_true==0, tf.float32)/t_pred[:,0]])
        y_pert = y_pred + tf.math.reduce_sum(tf.math.multiply(epsilons,h), axis=1)
        
        targeted_regularization = tf.reduce_sum(tf.square(y_true - y_pert))
        
        # Final
        loss = vanilla_loss + ratio * targeted_regularization
        return loss

    return tarreg_ATE_unbounded_domain_loss
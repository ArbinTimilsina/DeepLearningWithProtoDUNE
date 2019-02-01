import tensorflow as tf
from keras import backend as K

def dice_coefficient(y_true, y_pred):
    '''
    Sørensen–Dice coefficient is also known as the F1 score or Dice similarity coefficient (DSC).
    '''
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    smooth = 1.0
    return (2.0 * intersection + smooth)/(K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss():

    def loss(y_true, y_pred):
        return 1 - dice_coefficient(y_true, y_pred)

    return loss

def weighted_categorical_crossentropy(weights):
    """
    Weighted version of keras.objectives.categorical_crossentropy.
    Use this loss function for class balance.
    """
    # Convert weights to a constant tensor
    weights = tf.constant(weights, dtype=tf.float32)

    def loss(y_true, y_pred):
       # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)

        # Clip to prevent NaN's and Inf's
        _epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1.0 - _epsilon)

        # Do the loss calculation
        loss = y_true * tf.log(y_pred) * weights
        return -tf.reduce_sum(loss, axis=-1)

    return loss

def focal_loss(alpha=0.25, gamma=2.0):
    """
    alpha: Scale the focal weight with alpha.
    gamma: Take the power of the focal weight with gamma.
    """
    def loss(y_true, y_pred):
        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)

        # Clip to prevent NaN's and Inf's
        _epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1.0 - _epsilon)

        cross_entropy = y_true * tf.log(y_pred)

        # Do the focal loss calculation
        loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy

        return -tf.reduce_sum(loss, axis=-1)


    return loss

def intersection_over_union(y_true, y_pred, label):
    """
    Return the Intersection over Union (IoU) for a given label.
    """
    y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())

    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection

    return K.switch(K.equal(union, 0), 1.0, intersection / union)

def mean_iou(y_true, y_pred):
    """
    Return the mean Intersection over Union (IoU) score.
    """
    num_labels = K.int_shape(y_pred)[-1]

    total_iou = K.variable(0)
    for label in range(num_labels):
        # Note: WARNING:tensorflow:Variable += will be deprecated.
        total_iou = total_iou + intersection_over_union(y_true, y_pred, label)

    return total_iou / num_labels


import tensorflow as tf
import tensorflow_transform as tft

import constants

LABEL = constants.LABEL

def preprocessing_fn(inputs):

    outputs = dict()
    outputs['Age'] = inputs['Age']
    outputs['EstimatedSalary'] = inputs['EstimatedSalary']
    # converting the `Gender` into label encoded column.
    outputs['Gender'] = tf.cast(tf.equal(inputs['Gender'], 'male'), tf.int64)

    outputs[LABEL] = inputs[LABEL]

    return outputs

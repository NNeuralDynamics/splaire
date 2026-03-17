import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Conv1D, Cropping1D, BatchNormalization, add, Dropout
import tensorflow.keras.backend as kb
import numpy as np
from tensorflow.keras.regularizers import l2


def ResidualUnit(l, w, ar, dropout_rate=0.2):
    def f(input_node):
        bn1 = BatchNormalization()(input_node)
        act1 = Activation('relu')(bn1)
        drop1 = Dropout(dropout_rate)(act1)
        conv1 = Conv1D(
            filters=l,
            kernel_size=(w,),
            dilation_rate=(ar,),
            padding='same',
            kernel_initializer=tf.keras.initializers.GlorotUniform()
        )(drop1)
        bn2 = BatchNormalization()(conv1)
        act2 = Activation('relu')(bn2)
        drop2 = Dropout(dropout_rate)(act2)
        conv2 = Conv1D(
            l,
            kernel_size=(w,),
            dilation_rate=(ar,),
            padding='same',
            kernel_initializer=tf.keras.initializers.GlorotUniform()
        )(drop2)
        output_node = add([conv2, input_node])
        return output_node
    return f


def Splaire(L, W, AR, dropout_rate=0.2):
    assert len(W) == len(AR)

    CL = 2 * np.sum(AR * (W - 1))

    input0 = Input(shape=(None, 4))
    conv = Conv1D(
        L, 1,
        kernel_initializer=tf.keras.initializers.GlorotUniform()
    )(input0)

    drop_conv = Dropout(dropout_rate)(conv)

    skip = Conv1D(
        L, 1,
        kernel_initializer=tf.keras.initializers.GlorotUniform()
    )(drop_conv)

    for i in range(len(W)):
        conv = ResidualUnit(L, W[i], AR[i], dropout_rate)(conv)

        if ((i + 1) % 4 == 0) or ((i + 1) == len(W)):
            dense = Conv1D(
                L, 1,
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            )(conv)
            skip = add([skip, dense])

    skip = Cropping1D(int(CL / 2))(skip)

    bn_class = BatchNormalization()(skip)
    drop_class = Dropout(dropout_rate)(bn_class)

    output_class = Conv1D(
        3, 1,
        activation='softmax',
        name='classification_output',
        kernel_initializer=tf.keras.initializers.GlorotUniform()
    )(drop_class)

    bn_sse = BatchNormalization()(skip)
    drop_sse = Dropout(dropout_rate)(bn_sse)

    output_sse = Conv1D(
        1, 1,
        activation=None,
        name='regression_output',
        kernel_initializer=tf.keras.initializers.GlorotUniform()
    )(drop_sse)

    model = Model(inputs=input0, outputs=[output_class, output_sse])

    return model

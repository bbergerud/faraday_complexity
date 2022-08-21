"""
Basic CNN model architecture

classes
-------
classifier(channels, width)
    Simple model architecture that allows for easily adding
    additional layers through function calls. Assumes a channels
    last format.
"""

import tensorflow as tf
from typing import List, Optional, Union

class classifier:
    """
    Simple model architecture that allows for easily adding
    additional layers through function calls. Assumes a channels
    last format.

    Parameters
    ----------
    channels:int
        The number of input channels

    width: int, optional
        The size of the 1D tensor

    Methods
    -------
    activation(act, **kwargs)
        Adds an activation function to the model sequence.

    batchNorm(**kwargs)
        Adds an batch normalization operation to the model sequence.

    compile(loss, metrics, optimizer, weights)
        Adds a final dense and activation layer for the output and compiles
        the model.

    conv(filters, kernel_size, **kwargs)
        Adds a convolution operation to the model sequence.

    dense(neurons, **kwargs)
        Adds a dense operation to the model sequence.

    dropout(drop, **kwargs)
        Adds a dropout operation to the model sequence.

    flatten()
        Adds a flatten operation to the model sequence

    hidden(neurons, drop, activation, depth)
        Adds a set of hidden layers to the model sequence, defined as
        the sequence Dense -> Dropout -> Activation

    globalPooling(max_pool, **kwargs)
        Adds a global pooling layer to the model sequence

    pool(max_pool, **kwargs)
        Adds a pooling layer to the model sequence

    inception(conv, pool, max_pool, filters, stride, act)
        Adds an inception layer to the model sequence

    residual(kernel_size, act, batchNorm)
        Adds a residual unit to the model sequence.

    Examples
    --------
    import tensorflow as tf
    import numpy as np
    from possum.cnn import classifier

    optim = tf.keras.optimizers.SGD(learning_rate=0.001)

    model = classifier(channels=1)
    for i in range(5):
        model.conv(filters=8*(2**i))
        model.batchNorm()
        model.activation('relu')
        model.pool(max_pool=True)
    for i in range(2):
        model.inception(conv=[3,5], pool=[3], max_pool=True, act='relu', stride=1, filters=128)
    model.globalPooling(max_pool=True)
    model.dropout(0.3)
    model.compile(optimizer=optim)

    x = np.random.randn(5,101,1)
    y = model(x)
    print(y.shape)
    """
    def __init__(self, channels:int, width:Optional[int]=None):
        self.model = [tf.keras.layers.Input(shape=(width,channels))]

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def activation(self, act:str='relu', **kwargs):
        """
        Adds an activation function to the model sequence.

        Parameters
        ----------
        act:str
            The activation function to use

        **kwargs
            Addition arguments to pass into keras.layers.Activation
        """
        self.model.append(
            tf.keras.layers.Activation(activation=act, **kwargs)(self.model[-1])
        )

    def batchNorm(self, **kwargs):
        """
        Adds an batch normalization operation to the model sequence.

        Parameters
        ----------
        **kwargs
            Addition arguments to pass into keras.layers.BatchNormalization
        """
        self.model.append(
            tf.keras.layers.BatchNormalization(**kwargs)(self.model[-1])
        )

    def compile(self, 
        loss:Union[str,callable]='binary_crossentropy',
        metrics:List[Union[str,callable]]=['binary_accuracy'],
        optimizer:Union[str,tf.keras.optimizers.Optimizer]='rmsprop',
        weights:Optional[str]=None,
    ):
        """
        Adds a final dense and activation layer for the output and compiles
        the model.

        Parameters
        ----------
        loss : str, callable
            The loss function to apply
        
        metrics : list
            The set of metrics to apply

        optimizer : str, tf.keras.optimizers.Optimizer
            The model optimizer

        weights : str, optional
            The path location to stored weights for loading
        """
        self.model.append(tf.keras.layers.Dense(1)(self.model[-1]))
        self.model.append(tf.keras.layers.Activation('sigmoid')(self.model[-1]))
        self.model = tf.keras.models.Model(inputs=self.model[0], outputs=self.model[-1])

        if weights:
            self.model.load_weights(weights)

        self.model.compile(
            loss=loss,
            optimizer = optimizer,
            metrics=metrics
        )

    def conv(self, filters=32, kernel_size=3, **kwargs):
        """
        Adds a convolution operation to the model sequence.
   
        Parameters
        ----------
        filters : int
            The number of output channels/filters

        kernel_size : int
            The size of the convolution kernel

        **kwargs
            Addition arguments to pass into keras.layers.Conv1D
        """
        self.model.append(
            tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, **kwargs)(self.model[-1])
        )

    def dense(self, neurons, **kwargs):
        """
        Adds a dense operation to the model sequence.
   
        Parameters
        ----------
        neurons : int
            The number of neurons in the layer

        **kwargs
            Addition arguments to pass into keras.layers.Dense
        """
        self.model.append(tf.keras.layers.Dense(neurons, **kwargs))(self.model[-1])

    def dropout(self, drop, **kwargs):
        """
        Adds a dropout operation to the model sequence.
   
        Parameters
        ----------
        drop : float
            The droppout rate

        **kwargs
            Addition arguments to pass into keras.layers.Dropout
        """
        self.model.append(tf.keras.layers.Dropout(drop, **kwargs)(self.model[-1]))

    def flatten(self):
        """
        Adds a flatten operation to the model sequence
        """
        self.model.append(tf.keras.layers.Flatten()(self.model[-1]))

    def hidden(self, neurons:int, drop:float, activation:str, depth=1):
        """
        Adds a set of hidden layers to the model sequence, defined as
        the sequence Dense -> Dropout -> Activation

        Parameters
        ----------
        neurons : int
            The number of neurons in the layer

        drop : float
            The droppout rate

        act:str
            The activation function to use
        """
        for _ in range(depth):
            self.model.append(tf.keras.layers.Dense(neurons)(self.model[-1]))
            self.model.append(tf.keras.layers.Dropout(drop)(self.model[-1]))
            self.model.append(tf.keras.layers.Activation(activation)(self.model[-1]))

    def globalPooling(self, max_pool:bool=True, **kwargs):
        """
        Adds a global pooling layer to the model sequence

        Parameters
        ----------
        max_pool : bool
            Boolean indicating whether to apply max pooling (True)
            or average pooling (False)

        **kwargs
            Addition arguments to pass into keras.layers.Global[Max/Avg]Pooling1D
        """
        if max_pool:
            global_pool = tf.keras.layers.GlobalMaxPooling1D(**kwargs)(self.model[-1])
        else:
            global_pool = tf.keras.layers.GlobalAvgPooling1D(**kwargs)(self.model[-1])
        self.model.append(global_pool)

    def pool(self, max_pool=True, **kwargs):
        """
        Adds a pooling layer to the model sequence

        Parameters
        ----------
        max_pool : bool
            Boolean indicating whether to apply max pooling (True)
            or average pooling (False)

        **kwargs
            Addition arguments to pass into keras.layers.[Max/Avg]Pooling1D
        """
        if max_pool:
            pool = tf.keras.layers.MaxPooling1D(**kwargs)(self.model[-1])
        else:
            pool = tf.keras.layers.AvgPooling1D(**kwargs)(self.model[-1])
        self.model.append(pool)

    def inception(self, 
        conv:List[int],
        pool:List[int],
        max_pool:bool=True,
        filters:int=64,
        stride:int=1,
        act:str='relu'
    ):
        """
        Adds an inception layer to the model sequence

        Parameters
        ----------
        conv : List[int]
            A list containing a sequence of kernel sizes to use for the 
            convolutional operations

        pool : List[int]
            A list containing a sequence of kernel sizes to use for the
            pooling operations

        max_pool : bool
            Boolean indicating whether to apply max pooling (True)
            or average pooling (False)

        filters : int
            The number of output channels/filters

        stride : int
            The striding to use

        act : str
            The activation function to use
        """
        try: self.__inception += 1
        except: self.__inception = 1

        params = {
            "filters": filters,
            "strides": stride,
            "padding": 'same'
        }

        name = "layer{:d}/convl1x{:d}".format(self.__inception, 1)
        convl_1x1 = tf.keras.layers.Conv1D(kernel_size=1, name=name, **params)(self.model[-1])
        batch_1x1 = tf.keras.layers.BatchNormalization()(convl_1x1)
        activ_1x1 = tf.keras.layers.Activation(activation=act)(batch_1x1)

        model = [activ_1x1]
        for c in conv:
            convl_1x1 = tf.keras.layers.Conv1D(
                filters = filters // 2,
                kernel_size = 1,
                strides=1,
                padding='same'
            )(self.model[-1])
            batch_1x1 = tf.keras.layers.BatchNormalization()(convl_1x1)
            activ_1x1 = tf.keras.layers.Activation(activation=act)(batch_1x1)

            name = "layer{:d}/convl1x{:d}".format(self.__inception, c)
            convl_1xc = tf.keras.layers.Conv1D(kernel_size=c, name=name, **params)(activ_1x1)
            batch_1xc = tf.keras.layers.BatchNormalization()(convl_1xc)
            activ_1xc = tf.keras.layers.Activation(activation=act)(batch_1xc)

            model.append(activ_1xc)

        for p in pool:
            name = "layer{:d}/pool1x{:d}".format(self.__inception, p)
            params = {"pool_size": p, "strides": stride, "padding": 'same', "name": name}

            if max_pool:
                pool_1xc = tf.keras.layers.MaxPooling1D(**params)(self.model[-1])
            else:
                pool_1xc = tf.keras.layers.AvgPooling1D(**params)(self.model[-1])

            convl_1x1 = tf.keras.layers.Conv1D(
                filters = filters // 2,
                kernel_size = 1,
                strides=1,
                padding='same'
            )(pool_1xc)
            batch_1x1 = tf.keras.layers.BatchNormalization()(convl_1x1)
            activ_1x1 = tf.keras.layers.Activation(activation=act)(batch_1x1)
            model.append(activ_1x1)

        model = tf.keras.layers.concatenate(model)
        self.model.append(model)

    def residual(self, kernel_size:int=3, act:Optional[str]='relu', batchNorm:bool=True):
        """
        Adds a residual unit to the model sequence.

        Parameters
        ----------
        kernel_size : int
            The size of the convolution kernel

        act : str, optional
            The activation function to use. If None, then no activation
            function is added

        batchNorm : bool
            Bool indicating whether to add a batch normalization operation
            (True) or not (False)
        """
        model = []
        model.append(
            tf.keras.layers.Conv1D(
                filters=int(self.model[-1].shape[-1]),
                kernel_size=kernel_size,
                padding='same'
            )(self.model[-1])
        )

        if batchNorm:
            model.append(
                tf.keras.layers.BatchNormalization()(model[-1])
            )

        if act is not None:
            model.append(
                tf.keras.layers.Activation(act)(model[-1])
            )

        self.model.append(
            tf.keras.layers.add([self.model[-1], model[-1]])
        )
from functools import partial

from keras import backend as K
from tensorflow.python.keras.layers.noise import GaussianNoise
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.regularizers import l2, l1


class DummyHp():
    """A dummy class for hyperband, used when using parameters
    and not the lists for tuning"""
    def Choice(self, x, y):
        return y


class NetworkBuilder():

    def __init__(self, creator, n_input, input_shape, output_shape):
        self.creator = creator
        self.model = creator.model
        self.n_input = n_input
        self.input_shape = input_shape
        self.output_shape = output_shape

    def build_and_fit_model(
        self,
        hp=None,

        # Input Layer
        input_neurons=64,
        input_dropout_rate=0,
        use_input_regularizer=0,
        input_regularizer_penalty=0,

        # Hidden layer
        n_hidden_layers=1,
        hidden_layer_activation='relu',
        hidden_dropout_rate=.3,
        hidden_neurons=64,
        use_hidden_regularizer=0,
        hidden_regularizer_penalty=0,

        # Model fit
        #   Early Stopping
        use_early_stopping=True,
        monitor='val_loss',
        patience=5,
        #   Fit
        epochs=2000,
        batch_size=32,
        shuffle=False,

        # Other
        dummy_hp=False
                            ):

        if not hp and dummy_hp:
            hp = DummyHp()
        elif not hp and not dummy_hp:
            string = "No hp implemented, did you want dummy_hp=True?"
            raise AttributeError(string)

        # Possible clear old session
        try:
            del self.model
            K.clear_session()
        except AttributeError:
            pass

        # Model creation
        self.model = Sequential()

        # Input layer
        self.input_layer(
            hp,
            use_input_regularizer,
            input_regularizer_penalty,
            input_neurons,
            input_dropout_rate
            )

        # Hidden layers
        self.hidden_layers(
            hp,
            n_hidden_layers=1,
            hidden_layer_activation='relu',
            hidden_dropout_rate=.3,
            hidden_neurons=64,
            use_hidden_regularizer=0,
            hidden_regularizer_penalty=0,
            )
        self.output_layer(
            hp
            )

        self.make_fit(
            hp,
            # Model fit
            #   Early Stopping
            use_early_stopping=True,
            monitor='val_loss',
            patience=5,

            #   Fit
            epochs=2000,
            batch_size=32,
            shuffle=False,
        )

        return self.model

    def input_layer(
        self,
        hp,
        use_input_regularizer,
        input_regularizer_penalty,
        input_neurons,
        input_dropout_rate
                   ):
        #       Regularizer check
        _reg = None
        _use_reg = hp.Choice('use_input_regularizer',
                             use_input_regularizer)
        if _use_reg:
            _penalty = hp.Choice('input_regularizer_penalty',
                                 input_regularizer_penalty)
            if _use_reg > 1:
                _reg = l2(_penalty)
            else:
                _reg = l1(_penalty)

        #       Add input layer
        input_neurons = self.n_input*hp.Choice('input_neurons', input_neurons)
        self.model.add(LSTM(input_neurons,
                            input_shape=self.input_shape,
                            kernel_regularizer=_reg))

        #           Dropout layer
        input_dropout_rate = hp.Choice('input_dropout_rate',
                                       input_dropout_rate)
        if input_dropout_rate != 0:
            self.model.add(Dropout(input_dropout_rate))

        self.model.add(GaussianNoise(1))

    def hidden_layers(
        self,
        hp,
        use_hidden_regularizer,
        hidden_regularizer_penalty,
        hidden_dropout_rate,
        n_hidden_layers,
        hidden_neurons,
        hidden_layer_activation

                     ):
        #   Hidden layers
        #       Regularizer check
        _reg = None
        _use_reg = hp.Choice('use_hidden_regularizer',
                             use_hidden_regularizer)
        if _use_reg:
            _penalty = hp.Choice('hidden_regularizer_penalty',
                                 hidden_regularizer_penalty)
            if _use_reg > 1:
                _reg = l2(_penalty)
            else:
                _reg = l1(_penalty)

        #       Dropout check
        hidden_dropout_rate = hp.Choice('hidden_dropout_rate',
                                        hidden_dropout_rate)
        for i in range(hp.Choice('n_hidden_layers', n_hidden_layers)):
            self.model.add(
                Dense(hp.Choice('hidden_neurons',
                                hidden_neurons),
                      activation=hidden_layer_activation,
                      kernel_regularizer=_reg))

        #       Dropout layer
            if hidden_dropout_rate != 0:
                self.model.add(Dropout(hidden_dropout_rate))

    def output_layer(
        self,
        hp
                    ):
        #   Output Layer
        self.model.add(Dense(self.output_shape))

        #   Compile
        self.model.compile(optimizer='adam',
                           loss='mse')

    def make_fit(
        self,
        hp,
        # Model fit
        #   Early Stopping
        use_early_stopping=True,
        monitor='val_loss',
        patience=5,

        #   Fit
        epochs=2000,
        batch_size=32,
        shuffle=False,
                ):

        #   Define callbacks
        model_callbacks = []
        monitor = monitor
        patience = hp.Choice('patience', patience)
        use_early_stopping = hp.Choice('use_early_stopping',
                                       use_early_stopping)
        if use_early_stopping:
            model_callbacks.append(EarlyStopping(monitor=monitor,
                                                 patience=patience))

        # Fit partial
        self.model.fit = partial(
            self.model.fit,
            callbacks=model_callbacks,
            # epochs=hp.Choice('epochs', epochs),
            batch_size=hp.Choice('batch_size', batch_size),
            shuffle=shuffle
            )

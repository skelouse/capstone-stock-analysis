from functools import partial

from keras import backend as K
from tensorflow.python.keras.layers.noise import GaussianNoise
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.regularizers import l2, l1

from .sequential import CustomSequential


class DummyHp():
    """A dummy class for hyperband, used when using parameters
    and not the lists for tuning"""
    def Choice(self, x, y):
        return y


class NetworkBuilder():
    """
    Used by NetworkTuner and NetworkCreator
    for building and tuning the model

    Parameters
    ----------------------------------------
    creator[modeling.NetworkCreator]::
        -
    n_input(int)::
        - The number of timesteps to predict `tomorrow` with
    input_shape(tuple,)::
        - the input shape of the model
    output_shape(tuple,)::
        - the output shape of the model
    """
    # TODO add more parameters
    # TODO make a list out of parameters that are not
    # being tuned if dummy_hp=False
    # i.e
    # hidden_neurons = 3
    # -> hidden_neurons = [3]
    # may be able to remove dummy_hp quotient, and simply change everything
    # that is not a list to a list.  Then if len of all are one use dummy
    # else if they are not all one use the provided hp.
    def __init__(self, creator, n_input, input_shape, output_shape):
        self.creator = creator
        self.model = creator.model
        self.n_input = n_input
        self.input_shape = input_shape
        self.output_shape = output_shape

    def build_and_fit_model(
        self,
        hp=None,
        # Data
        n_days=1,

        # Input Layer
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
        """
        Parameters
        ----------------------------------------
        hp=None,
        dummy_hp=False(bool)::
            - whether to use a dummy_hp or not for simply building the
            model or tuning it.  Will return an error if the model is not
            tuning and dummy_hp is False

        Data Params
            ------------------------------------
        n_days=1,

        Input Layer Params
            ------------------------------------
        input_neurons=64
        input_dropout_rate=0
        use_input_regularizer=0
        input_regularizer_penalty=0

        Hidden Layer Params
            ------------------------------------
        n_hidden_layers=1
        hidden_layer_activation='relu'
        hidden_dropout_rate=.3
        hidden_neurons=64
        use_hidden_regularizer=0
        hidden_regularizer_penalty=0

        Early Stopping Params
            ------------------------------------
        use_early_stopping=True,
        monitor='val_loss',
        patience=5,

        Model Fit Params
            ------------------------------------
        epochs=2000,
        batch_size=32,
        shuffle=False,

        Returns
        ----------------------------------------
        self.model(tensorflow.keras.models.Sequential)
        OR
        self.model(.modeling.CustomSequantial) when tuning
            - built using parameters
            - fit function is functools.partial with
              defined fit arguments already plugged
        """

        self.hp = hp
        if not hp and dummy_hp:
            self.hp = DummyHp()
        elif not hp and not dummy_hp:
            string = "No hp implemented, did you want dummy_hp=True?"
            raise AttributeError(string)

        def clear_sess():
            """Used for freeing ram by clearing model and keras session"""
            try:
                del self.model
                K.clear_session()
            except AttributeError:
                pass

        # Model creation
        # Check if building model from parameters or tuning
        if isinstance(self.hp, DummyHp):
            clear_sess()
            self.model = Sequential()
        else:
            clear_sess()
            n_days = self.hp.Choice('n_days',
                                    n_days)
            self.model = CustomSequential(self.creator.k_folds, n_days)

        # Input layer
        self.input_layer(
            use_input_regularizer,
            input_regularizer_penalty,
            input_dropout_rate
            )

        # Hidden layers
        self.hidden_layers(
            n_hidden_layers,
            hidden_layer_activation,
            hidden_dropout_rate,
            hidden_neurons,
            use_hidden_regularizer,
            hidden_regularizer_penalty,
            )

        self.output_layer(
            )

        self.make_fit(
            # Model fit
            #   Early Stopping
            use_early_stopping,
            monitor,
            patience,

            #   Fit
            epochs,
            batch_size,
            shuffle,
        )
        return self.model

    def input_layer(
        self,
        use_input_regularizer,
        input_regularizer_penalty,
        input_dropout_rate
                   ):
        #       Regularizer check
        _reg = None
        _use_reg = self.hp.Choice('use_input_regularizer',
                                  use_input_regularizer)
        if _use_reg:
            _penalty = self.hp.Choice('input_regularizer_penalty',
                                      input_regularizer_penalty)
            if _use_reg > 1:
                _reg = l2(_penalty)
            else:
                _reg = l1(_penalty)

        #       Add input layer
        input_neurons = self.creator.X_n_features
        self.model.add(LSTM(input_neurons,
                            input_shape=self.input_shape,
                            kernel_regularizer=_reg))

        #           Dropout layer
        input_dropout_rate = self.hp.Choice('input_dropout_rate',
                                            input_dropout_rate)
        if input_dropout_rate != 0:
            self.model.add(Dropout(input_dropout_rate))

        self.model.add(GaussianNoise(1))

    def hidden_layers(
        self,
        n_hidden_layers,
        hidden_layer_activation,
        hidden_dropout_rate,
        hidden_neurons,
        use_hidden_regularizer,
        hidden_regularizer_penalty
                     ):
        #   Hidden layers
        #       Regularizer check
        _reg = None
        _use_reg = self.hp.Choice('use_hidden_regularizer',
                                  use_hidden_regularizer)
        if _use_reg:
            print("\n\nhidden_regularizer_penalty\n")
            print(hidden_regularizer_penalty)

            _penalty = self.hp.Choice('hidden_regularizer_penalty',
                                      hidden_regularizer_penalty)
            if _use_reg > 1:
                _reg = l2(_penalty)
            else:
                _reg = l1(_penalty)

        #       Dropout check
        hidden_dropout_rate = self.hp.Choice('hidden_dropout_rate',
                                             hidden_dropout_rate)
        for i in range(self.hp.Choice('n_hidden_layers', n_hidden_layers)):
            self.model.add(
                Dense(self.hp.Choice('hidden_neurons',
                                     hidden_neurons),
                      activation=hidden_layer_activation,
                      kernel_regularizer=_reg))

        #       Dropout layer
            if hidden_dropout_rate != 0:
                self.model.add(Dropout(hidden_dropout_rate))

    def output_layer(
        self
                    ):
        #   Output Layer
        self.model.add(Dense(self.output_shape))

        #   Compile
        self.model.compile(optimizer='adam',
                           loss='mse')

    def make_fit(
        self,
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
        patience = self.hp.Choice('patience', patience)
        use_early_stopping = self.hp.Choice('use_early_stopping',
                                            use_early_stopping)
        if use_early_stopping:
            model_callbacks.append(EarlyStopping(monitor=monitor,
                                                 patience=patience))

        # Fit partial
        if isinstance(self.hp, DummyHp):
            self.model.fit = partial(
                self.model.fit,
                callbacks=model_callbacks,
                # epochs=self.hp.Choice('epochs', epochs),
                batch_size=self.hp.Choice('batch_size', batch_size),
                shuffle=shuffle
                )
        else:
            self.model.fit = partial(
                self.model.fit,
                callbacks=model_callbacks,
                # epochs=self.hp.Choice('epochs', epochs),
                batch_size=self.hp.Choice('batch_size', batch_size),
                shuffle=shuffle,
                n_days=1
                # TODO remove hardwire n_days
                )

self.model = Sequential()
self.model.add(LSTM(64, input_shape=input_shape))
self.model.add(Dropout(.3))
for i in range(1):
    self.model.add(Dense(64, activation='relu'))
    self.model.add(Dropout(.1))
self.model.add(Dense(len(self.y_cols)))

self.model.compile(optimizer='adam', loss='mse')

earlystopping = EarlyStopping(monitor='val_loss', patience=25)
history = self.model.fit(self.train_data_gen,
                    epochs=2000, batch_size=64,
                    validation_data=(self.test_data_gen),
                    verbose=2, shuffle=False,
                    callbacks=[earlystopping])
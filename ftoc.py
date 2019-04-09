import tensorflow as tf
import numpy as np

print('hello')


celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahren_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

for i,c in enumerate(celsius_q):
    print(" {} c = {} f".format(c, fahren_a[i]))


# layer 0
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

model = tf.keras.Sequential([l0])


model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(0.1))

history = model.fit(celsius_q, fahren_a, epochs = 500, verbose = False)

print("Finished training the model")

print(model.predict([100.0]))
import numpy as np

import tensorflow as tf

print(tf.config.list_physical_devices("GPU"))
print(tf.test.is_built_with_gpu_support())

import tensorflow.keras as keras


# inputs = Input(shape=(784,))
# # Just for demonstration purposes.
# img_inputs = keras.Input(shape=(32, 32, 3))
# dense = keras.layers.Dense(64, activation="relu")
# x = dense(inputs)
# x = keras.layers.Dense(64, activation="relu")(x)
# outputs = keras.layers.Dense(10)(x)
# model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# x_train = x_train.reshape(60000, 784).astype("float32") / 255
# x_test = x_test.reshape(10000, 784).astype("float32") / 255

# model.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer=keras.optimizers.RMSprop(),
#     metrics=["accuracy"],
# )

# history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

# test_scores = model.evaluate(x_test, y_test, verbose=2)
# print("Test loss:", test_scores[0])
# print("Test accuracy:", test_scores[1])


input_shape = (1000, 6)
x = tf.random.normal(input_shape)
output_shape = (1000, 10)
y = tf.random.normal(output_shape)

x = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [1, -1, -1, -1, -1, -1],
        [1, -1, -1, 1, -1, -1],
        [1, 0, -1, 1, -1, -1],
        [1, -1, -1, 1, -1, 0],
        [1, 0, 0, 0, 0, 0],
    ]
)

y = np.array(
    [
        [0.01, 0.005, 0.005, 0.005, 0.005, 0.001, 0.001, 0.001, 0.001, 0.001],
        [0.99, 0.16, 0.34, 0.66, 0.495, 0.165, 0.15, 0.335, 0.65, 0.16],
        [0.995, 0.1, 0.1683, 0.99, 0.75, 0.275, 0.09, 0.16, 0.74, 0.27],
        [0.997, 0.001, 0.17, 0.995, 0.76, 0.28, 0.001, 0.17, 0.75, 0.28],
        [0.993, 0.1, 0.17, 0.98, 0.76, 0.005, 0.09, 0.165, 0.75, 0.001],
        [0.9, 0.005, 0.01, 0.01, 0.01, 0.005, 0.001, 0.005, 0.005, 0.005],
    ]
)

dlnn = keras.Sequential(
    [
        keras.Input(shape=input_shape[1]),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(10, activation="sigmoid"),
    ]
)
print(dlnn.output_shape)


dlnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
dlnn.fit(x, y, epochs=1, batch_size=32)

print(dlnn.predict([[1, 0, 0, 1, -1, -1]]))

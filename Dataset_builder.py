from keras.datasets import mnist
import tensorflow as tf

class Dataset_builder:
    def __init__(self):
        (self.x_train, _), (_, _) = mnist.load_data()
        pass

    def get_dataset(self, batch_size):
        self.x_train = tf.cast(self.x_train, dtype=tf.float32)
        self.x_train = tf.expand_dims(self.x_train, axis = -1)
        self.x_train = tf.data.Dataset.from_tensor_slices(self.x_train).map(self.map_dataset).batch(batch_size=batch_size, drop_remainder=True).shuffle(buffer_size=batch_size)
        return self.x_train

    def map_dataset(self, datapoint):
        datapoint = (datapoint-127.5)/127.5
        return datapoint

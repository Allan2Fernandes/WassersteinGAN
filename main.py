import tensorflow as tf
import Dataset_builder
import GAN

#Constants
batch_size = 256
noise_dimensions=128
learning_rate = 0.0001
gp_weight = 10
epochs = 200

dataset_builder = Dataset_builder.Dataset_builder()
dataset = dataset_builder.get_dataset(batch_size=batch_size)


network = GAN.GAN(dataset, noise_dimensions=noise_dimensions)

network.build_critic()
network.build_generator()

network.initialize_optimizers(learning_rate=learning_rate)
network.initialize_loss_function()

network.custom_train(epochs=epochs)





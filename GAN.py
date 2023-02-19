import tensorflow as tf
import matplotlib.pyplot as plt
import time
from keras.layers import Dense, Conv2D, Input, BatchNormalization, LeakyReLU, Flatten, Reshape, Conv2DTranspose, Lambda
from keras.optimizers import Adam

class GAN:
    def __init__(self, dataset, noise_dimensions):
        self.dataset = dataset
        self.noise_dimensions = noise_dimensions
        pass

    def generate_image(self):
        noise = tf.random.normal(shape=[1, self.noise_dimensions])
        fake_image = self.generator_model(noise)
        plt.imshow(fake_image[0])
        plt.show()
        pass

    def build_generator(self):
        input_layer = Input(shape=(self.noise_dimensions))
        dense_1 = Dense(units = 7*7*256)(input_layer)
        reshape_layer = Reshape(target_shape=(7,7,256))(dense_1)
        x = BatchNormalization()(reshape_layer)
        x = LeakyReLU()(x)
        first_block = self.generator_block(x, filters=128)
        second_block = self.generator_block(first_block, filters=1, final_block=True)

        self.generator_model = tf.keras.Model(inputs = input_layer, outputs=second_block)
        self.generator_model.summary()
        pass

    def generator_block(self, input_layer, filters, kernel_size=4, strides=2, final_block=False, alpha=0.2):
        if not final_block:
            x = Conv2DTranspose(kernel_size=kernel_size, filters=filters, strides=strides, padding = 'same', use_bias=False)(input_layer)
            x = BatchNormalization()(x)
            x = Lambda(lambda x: tf.keras.activations.selu(x))(x)
        else:
            x = Conv2DTranspose(kernel_size=kernel_size, filters=filters, strides=strides, activation='tanh', padding='same', use_bias=False)(input_layer)
            pass

        return x


    def build_critic(self):
        input_layer = Input(shape=(28, 28, 1))
        first_block = self.critic_block(input_layer=input_layer, filters=64)
        print(first_block.shape)
        second_block = self.critic_block(input_layer=first_block, filters=128)
        print(second_block.shape)
        last_block = self.critic_block(input_layer=second_block, filters=256, final_block=True)

        self.critic_model = tf.keras.Model(inputs=input_layer, outputs = last_block)
        self.critic_model.summary()
        pass

    def critic_block(self, input_layer, filters, kernel_size=4, strides=2, final_block=False, alpha= 0.2):
        x = Conv2D(kernel_size=kernel_size, strides=strides, padding='same', filters=filters, use_bias=False)(input_layer)
        if not final_block:
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=alpha)(x)
        else:
            x = LeakyReLU(alpha=alpha)(x)
            x = Flatten()(x)
            x = Dense(units=1)(x)
            pass
        return x
    def get_generator(self):
        return self.generator_model

    def get_discriminator(self):
        return self.critic_model

    def initialize_optimizers(self, learning_rate):
        self.gen_optimizer = Adam(learning_rate=learning_rate)
        self.critic_optimizer = Adam(learning_rate=learning_rate)
        pass

    def initialize_loss_function(self): #Unused
        self.loss_function = tf.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        pass

    def get_gradient(self, critic, real_images, fake_images, weight):
        mixed_images = real_images*weight + fake_images*(1*weight)
        with tf.GradientTape() as tape:
            tape.watch(mixed_images)
            mixed_scores = critic(mixed_images)
            pass
        gradient = tape.gradient(mixed_scores, mixed_images)
        return gradient

    def gradient_penalty(self, gradient, batch_size):
        gradient = tf.reshape(gradient, [batch_size, -1])
        gradient_norm = tf.norm(gradient)
        penalty = tf.reduce_mean((gradient_norm-1)**2)
        return penalty

    def get_crit_loss(self, crit_fake_prediction, crit_real_prediction, gradient_penalty, c_lambda):
        total_crit_loss = -crit_real_prediction + crit_fake_prediction + c_lambda*gradient_penalty
        return total_crit_loss

    def get_gen_loss(self, crit_fake_pred):
        gen_loss = -tf.reduce_mean(crit_fake_pred)
        return gen_loss

    def custom_train(self, epochs, c_lambda=10):
        for epoch in range(epochs):
            for step, real_images in enumerate(self.dataset):
                #Get the batch size
                batch_size = real_images.shape[0]
                with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
                    #Create noise
                    noise_vector = tf.random.normal([batch_size, self.noise_dimensions], 0.0, 1.0)
                    #Pass it through the generator to get fake images
                    fake_images = self.generator_model(noise_vector)

                    #Get the gradient penalty
                    #Get a random broadcastable weight
                    weight = tf.random.normal(shape=[batch_size, 1, 1, 1])
                    #Get the score gradient
                    disc_score_gradient = self.get_gradient(critic=self.critic_model, real_images=real_images, fake_images=fake_images, weight=weight)
                    #Use the score gradient to get the gradient penalty
                    gradient_penalty = self.gradient_penalty(gradient=disc_score_gradient, batch_size=batch_size)


                    # Pass it through the discriminator to get fake predictions
                    fake_image_predictions = self.critic_model(fake_images)
                    #Pass the real images to get real predictions
                    real_image_predictions = self.critic_model(real_images)
                    #Use the predictions to calculate the total crit loss
                    total_crit_loss = self.get_crit_loss(crit_fake_prediction=fake_image_predictions, crit_real_prediction=real_image_predictions, gradient_penalty=gradient_penalty, c_lambda=c_lambda)


                    total_gen_loss = self.get_gen_loss(crit_fake_pred=fake_image_predictions)
                    pass
                # Train the discriminator
                #Differentiate disc loss wrt discriminator variables
                disc_gradient = disc_tape.gradient(total_crit_loss, self.critic_model.trainable_variables)
                #Gradient descent
                self.critic_optimizer.apply_gradients(zip(disc_gradient, self.critic_model.trainable_variables))
                # Train the generator
                gen_gradient = gen_tape.gradient(total_gen_loss, self.generator_model.trainable_variables)
                self.gen_optimizer.apply_gradients(zip(gen_gradient, self.generator_model.trainable_variables))
                print("Epoch: {0} Step: {1} || Discriminator loss = {2} || Generator loss = {3}".format(epoch, step, tf.reduce_sum(total_crit_loss), tf.reduce_sum(total_gen_loss)))
                pass #End of step
            self.generate_image()
            pass #End of epoch
        pass
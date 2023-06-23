from tensorflow import keras as K
from tqdm import trange
import tensorflow as tf
import numpy as np


class TrainingModel(K.Sequential):
    def __init__(self, weights, args, num_states, lr_optim):
        super().__init__()
        self.optimizer = K.optimizers.Adam(lr_optim)  # type: ignore
        self.num_states = num_states
        self.batch_size = args.batch_size
        self.indexes = tf.range(args.num_samples)

        if weights:
            # ,weights[14].numpy(),weights[16].numpy(),weights[18].numpy(),weights[20].numpy(),weights[22].numpy(),weights[24].numpy(),weights[26].numpy(),weights[28].numpy(),weights[30].numpy(),weights[32].numpy()
            kernel_weights = weights[0].numpy(), weights[2].numpy(), weights[4].numpy()  # , weights[6].numpy(), weights[8].numpy(), weights[10].numpy(), weights[12].numpy()
            # ,weights[13].numpy(),weights[15].numpy(),weights[17].numpy(),weights[19].numpy(),weights[21].numpy(),weights[23].numpy(),weights[25].numpy(),weights[27].numpy(),weights[29].numpy(),weights[31].numpy()
            bias_weights = weights[1].numpy(), weights[3].numpy(), weights[5].numpy()  # , weights[7].numpy(), weights[9].numpy(), weights[11].numpy(), weights[13].numpy()

            # Hidden layers
            for layer in range(args.num_hidden_layers):
                self.add(K.layers.Dense(args.num_neurons, activation=args.activation_function, input_dim=num_states, kernel_initializer=K.initializers.Constant(value=kernel_weights[layer]),
                                        bias_initializer=K.initializers.Constant(value=bias_weights[layer])))

            # Output layer
            self.add(K.layers.Dense(args.model_output_size, kernel_initializer=K.initializers.Constant(
                value=kernel_weights[args.num_hidden_layers]), bias_initializer=K.initializers.Constant(value=bias_weights[args.num_hidden_layers]), activation=args.activation_function_output))

        else:
            # Hidden layers
            for layer in range(args.num_hidden_layers):
                self.add(K.layers.Dense(args.num_neurons, activation=args.activation_function, input_dim=num_states))

            # Output layer
            self.add(K.layers.Dense(args.model_output_size, bias_initializer=K.initializers.Constant(value=args.initial_guess), activation=args.activation_function_output))

        self.compile(optimizer = self.optimizer, loss = 'mse',steps_per_execution = 10000)
    # def train(self, train_data, number_epochs):
    #     losses_primes = []
    #     weights = self.trainable_variables

    #     gradients_of_primes = self.training_start(train_data)
    #     self.optimizer.apply_gradients(zip(gradients_of_primes, weights))

    #     for _ in trange(number_epochs):
    #         mean_loss_prime = self.training_step(train_data)
    #         losses_primes.append(mean_loss_prime)

    #     print(f'Done...\nTrain mean loss: {np.mean(np.array(losses_primes)[-2000:])}')
    #     return losses_primes

    # @tf.function(reduce_retracing=True)
    # def objective_neuralnet(self, data):
    #     random_indexes = tf.random.shuffle(self.indexes)[:self.batch_size]
    #     data_batch = tf.gather(data, random_indexes)

    #     states_batch = data_batch[:, :self.num_states]
    #     value_prime_optimal_batch = tf.expand_dims(data_batch[:, -1], axis=1)
    #     error_value_prime_neuralnet = tf.reduce_mean(tf.square(value_prime_optimal_batch - self(states_batch)))

    #     return error_value_prime_neuralnet

    # @tf.function
    # def training_step(self, data):
    #     with tf.GradientTape() as tape:
    #         value_prime_loss = self.objective_neuralnet(data)
    #     value_prime_gradients = tape.gradient(value_prime_loss, self.trainable_variables)
    #     self.optimizer.apply_gradients(zip(value_prime_gradients, self.trainable_variables))
    #     return value_prime_loss

    # @tf.function()
    # def training_start(self, data):
    #     with tf.GradientTape() as tape:
    #         value_prime_loss = self.objective_neuralnet(data)
    #     value_prime_gradients = tape.gradient(value_prime_loss, self.trainable_variables)
    #     return value_prime_gradients
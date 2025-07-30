library(keras)
library(tensorflow)
library(tidyverse)

##################################################################
# Step 0: Set the random seed and specify parameters
##################################################################

tensorflow::set_random_seed(33)

random_noise_dim = 4
output_dim = 2
hidden_units = 16
batch_size = 32
epochs = 10000

##################################################################
# Step 1. Define helper functions
##################################################################

# Generate data from a normal distribution that has a mean of 5.
# The generator will attempt to replicate this.
# The result is a matrix with n rows and output_dim columns.
generate_real_data = function(n) {
  matrix(rnorm(n * output_dim, mean = 5), nrow = n)
}

# Generate data from a uniform distribution where values have a
# minimum of -1 and a maximum of 1.
# The result is a matrix with n rows and random_noise_diff columns.
generate_fake_data = function(n) {
  matrix(runif(n * random_noise_dim, min = -1, max = 1), nrow = n)
}

##################################################################
# Step 2. Define the generator
##################################################################

generator = keras_model_sequential() %>%
  layer_dense(units = hidden_units, input_shape = c(random_noise_dim)) %>%
  layer_batch_normalization() %>% # This piece was not described in the book, but we added it to try to stabilize the training process.
  layer_activation_leaky_relu(alpha = 0.1) %>%
  layer_dense(units = output_dim, activation = "linear")

# Compile the generator individually.
generator %>% compile(
  optimizer = "adam",
  loss = "mse"
)

##################################################################
# Step 3. Define and compile the discriminator
##################################################################

discriminator = keras_model_sequential() %>%
  layer_dense(units = hidden_units, input_shape = c(output_dim), activation = layer_activation_leaky_relu(alpha = 0.1)) %>%
  layer_dense(units = hidden_units, activation = layer_activation_leaky_relu(alpha = 0.1)) %>%
  layer_dense(units = 1, activation = "sigmoid")  # Outputs a probability (real or fake)

# Compile the discriminator individually.
discriminator %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy"
)

##################################################################
# Step 4. Define and compile the GAN (generator + discriminator)
##################################################################

# Freeze discriminator weights while training the generator.
discriminator %>% freeze_weights()

# Connect the generator with the discriminator.
gan = keras_model_sequential() %>%
  generator %>%
  discriminator

# Compile the GAN using the frozen discriminator weights.
gan %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy"
)

# Now we unfreeze the weights for the discriminator.
# I'm not 100% sure whether this step is necessary or if Keras handles it
#   automatically.
discriminator %>% unfreeze_weights()

# Recompile the discriminator with the unfrozen weights.
discriminator %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy"
)

##################################################################
# Step 5. Train the GAN
##################################################################

# For each epoch, alternate between training the discriminator on
# real and fake data, and train the generator using the combined GAN model.
# The book describes 4 distinct steps. Those are implied rather than
# implicit in the code.
for (epoch in 1:epochs) {
  # Generate real samples
  real_data = generate_real_data(batch_size)
  
  # Generate fake samples
  fake_data = generator %>%
    predict(generate_fake_data(batch_size), verbose = 0)
  
  # Combine real and fake data
  x = rbind(fake_data, real_data)
  
  # Create labels for the real and fake data. It uses 0 for fake.
  # However, for real it uses a technique called "label smoothing" that assigns
  #   a random value between 0.8 and 1.0.
  y = c(rep(0, batch_size), runif(batch_size, 0.8, 1.0))  # Apply label smoothing
  
  # Train the discriminator.
  discriminator_loss = discriminator %>%
    train_on_batch(x, y)
  
  # Train the generator (via the GAN model).
  noise = generate_fake_data(batch_size)
  y_gan = rep(1, batch_size)  # Trick GAN into thinking fake data is real
  
  gan_loss = gan %>%
    train_on_batch(noise, y_gan)
  
  # Print progress every 100 epochs
  if (epoch %% 100 == 0) {
    print(sprintf("Epoch: %d, Discriminator loss: %.4f, Generator loss: %.4f", epoch, discriminator_loss, gan_loss))
  }
}

##################################################################
# Step 5. Generate new data
##################################################################

num_test_samples = 10000

# Generate random noise samples for testing the model
noise = generate_fake_data(num_test_samples)

# Transform the noise using the GAN
generated_data = generator %>%
  predict(noise, verbose = 0) %>%
  as_tibble() %>%
  dplyr::rename(x = V1, y = V2)

##################################################################
# Step 6. Plot the generated data.
##################################################################

# Create a histogram of the x values.
ggplot(generated_data, aes(x = x)) +
  geom_histogram() +
  theme_bw()

# Create a histogram of the y values.
ggplot(generated_data, aes(x = y)) +
  geom_histogram() +
  theme_bw()

# Create a scatter plot of the x and y values.
ggplot(generated_data, aes(x = x, y = y)) +
  geom_point() +
  theme_bw()

# Save the last graph to a file so it can be shared.
ggsave("Chapter22_Scatterplot.png")
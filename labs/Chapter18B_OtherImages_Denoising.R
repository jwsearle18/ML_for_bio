library(keras)
library(tidyverse)

# https://www.kaggle.com/datasets/nirmalsankalana/rice-leaf-disease-image

tensorflow::set_random_seed(0)

img_width = 120
img_height = 120
channels = 3
batch_size = 32

image_data_gen = image_data_generator(rescale = 1/255)

root_dir = "~/Desktop/bio564/rice_leaf_disease"
clean_dir = paste0(root_dir, "/Blast")

# Function to load images from a directory into an array
load_images_to_array = function(directory, img_width, img_height) {
  # List all image files in the directory
  image_files = list.files(directory, full.names = TRUE)
  
  # Initialize an array to store the images
  num_images = length(image_files)
  image_array = array(0, dim = c(num_images, img_height, img_width, channels))
  
  # Load and preprocess each image
  for (i in seq_along(image_files)) {
    img_path = image_files[i]
    img = image_load(img_path, target_size = c(img_height, img_width))
    img_array = image_to_array(img) / 255  # Normalize pixel values to [0, 1]
    image_array[i,,,] = img_array
  }
  
  return(image_array)
}

clean_images = load_images_to_array(clean_dir, img_width, img_height)

# Choose the first 1000 images as the training set.
clean_images_train = clean_images[1:1000,,,]

# Retrieve the remaining images for the test set.
clean_images_test = clean_images[1001:1440,,,]

add_noise = function(images, noise_factor) {
  noisy_images = images + noise_factor * array(rnorm(length(images)), dim = dim(images))
  noisy_images = pmax(pmin(noisy_images, 1), 0)  # Clip values to [0, 1]
  return(noisy_images)
}

noisy_images = add_noise(clean_images, noise_factor = 0.2)

# Separate these into training and test set.
noisy_images_train = noisy_images[1:1000,,,]
noisy_images_test = noisy_images[1001:1440,,,]

input_img = layer_input(shape = c(img_height, img_width, 3))

# Encoder
encoded = input_img %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), strides = c(2, 2), activation = "leaky_relu", padding = "same") %>%
  layer_batch_normalization() %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), strides = c(2, 2), activation = "leaky_relu", padding = "same") %>%
  layer_batch_normalization()

# Bottleneck
bottleneck = encoded %>%
  layer_conv_2d(filters = 256, kernel_size = c(3, 3), strides = c(2, 2), activation = "leaky_relu", padding = "same") %>%
  layer_batch_normalization()

# Decoder
decoded = bottleneck %>%
  layer_conv_2d_transpose(filters = 64, kernel_size = c(3, 3), strides = c(2, 2), activation = "leaky_relu", padding = "same") %>%
  layer_conv_2d_transpose(filters = 32, kernel_size = c(3, 3), strides = c(2, 2), activation = "leaky_relu", padding = "same") %>%
  layer_conv_2d_transpose(filters = 32, kernel_size = c(3, 3), strides = c(2, 2), activation = "leaky_relu", padding = "same") %>%
  layer_conv_2d_transpose(filters = 3, kernel_size = c(3, 3), activation = "sigmoid", padding = "same")

# Define the autoencoder model
autoencoder = keras_model(inputs = input_img, outputs = decoded)

# Compile the model
autoencoder %>% compile(optimizer = "adam", loss = "mse")

history = autoencoder %>% fit(
  x = noisy_images_train,
  y = clean_images_train,
  epochs = 50,
  batch_size = batch_size,
  validation_split = 0.2
)

# Predict on noisy test data
denoised_images_test = autoencoder %>%
  predict(noisy_images_test)

# Convert the history object to a data frame
loss_df = data.frame(
  epoch = 1:length(history$metrics$loss),
  training_loss = history$metrics$loss,
  validation_loss = history$metrics$val_loss
)

# Reshape data for easier plotting with ggplot2
loss_df_long = loss_df %>%
  pivot_longer(cols = c(training_loss, validation_loss), names_to = "type", values_to = "loss")

# Plot with ggplot2
ggplot(loss_df_long, aes(x = epoch, y = loss, color = type)) +
  geom_line(linewidth = 1) +
  labs(
    title = "Training and Validation Loss Over Epochs",
    x = "Epoch",
    y = "Loss",
    color = ""
  ) +
  scale_color_manual(values = c("training_loss" = "blue", "validation_loss" = "red"), 
                     labels = c("Training Loss", "Validation Loss")) +
  theme_minimal()


# Plotting some examples
par(mfrow = c(3, 3), mar = c(0, 0, 0, 0))

for (i in 1:3) {
  # Original noisy image
  plot(as.raster(clean_images_test[i,,,]), main = "Clean")
  
  # Original noisy image
  plot(as.raster(noisy_images_test[i,,,]), main = "Noisy")
  
  # Denoised image
  plot(as.raster(denoised_images_test[i,,,]), main = "Denoised")
}
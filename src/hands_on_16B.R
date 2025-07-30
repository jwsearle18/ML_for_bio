library(tidyverse)
library(tidymodels)
library(keras)

tensorflow::set_random_seed(0)

target_image_width = 56
target_image_height = 56
batch_size = 32

dir_path = "fungi/"

train_dir = paste0(dir_path, "train/")
valid_dir = paste0(dir_path, "valid/")
test_dir = paste0(dir_path, "test/")

train_image_gen = image_data_generator(
  rescale = 1/255,  # Normalizes pixel values to [0, 1]
  width_shift_range = 0.2,   # Horizontal shift
  height_shift_range = 0.2,  # Vertical shift
  zoom_range = 1.8,          # Random zoom
  shear_range = 0.2,         # Shear transformation
  horizontal_flip = TRUE,    # Random horizontal flip
  vertical_flip = TRUE,      # Random vertical flip
  rotation_range = 20
)

valid_image_gen = image_data_generator(
  rescale = 1/255
)

test_image_gen = image_data_generator(
  rescale = 1/255
)

train_generator = flow_images_from_directory(
  generator = train_image_gen,
  directory = train_dir,
  target_size = c(target_image_height, target_image_width),
  batch_size = batch_size,
  class_mode = "categorical",
  color_mode = "rgb"
)

test_generator = flow_images_from_directory(
  generator = test_image_gen,
  directory = test_dir,
  target_size = c(target_image_height, target_image_width),
  batch_size = batch_size,
  class_mode = "categorical",
  color_mode = "rgb"
)

valid_generator = flow_images_from_directory(
  generator = valid_image_gen,
  directory = valid_dir,
  target_size = c(target_image_height, target_image_width),
  batch_size = batch_size,
  class_mode = "categorical",
  color_mode = "rgb"
)

output_dim = length(unique(train_generator$class_indices))

model = keras_model_sequential() %>%
  layer_conv_2d(
    kernel_size = c(3, 3),
    strides = c(2, 2),
    filters = 32,
    activation = 'relu',
    padding = 'valid',
    input_shape = c(target_image_height, target_image_width, 3)) %>%
  # layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(
    kernel_size = c(3, 3),
    strides = c(3, 3),
    filters = 64,
    activation = 'relu',
    padding = 'valid') %>%
  # layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(
    kernel_size = c(3, 3),
    # strides = c(3, 3),
    filters = 128,
    activation = 'relu',
    padding = 'valid') %>%
  layer_max_pooling_2d(pool_size = c(3, 3)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = output_dim, activation = 'softmax')

compile(model, optimizer = 'adam',
        loss = "categorical_crossentropy",
        metrics = c('accuracy'))

history = model %>% fit(
  train_generator,
  epochs = 5,
  validation_data = valid_generator
)

evaluation <- model %>%
  evaluate(test_generator)

print(evaluation['accuracy'])
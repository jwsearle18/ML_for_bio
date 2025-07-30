library(tidyverse)
library(keras)

tensorflow::set_random_seed(0)

target_image_width = 64
target_image_height = 32
batch_size = 32

dir_path = "dental/"

train_dir = paste0(dir_path, "train/")
valid_dir = paste0(dir_path, "valid/")
test_dir = paste0(dir_path, "test/")

train_annotations = read_csv(paste0(train_dir, "_annotations.csv"))
test_annotations = read_csv(paste0(test_dir, "_annotations.csv"))
valid_annotations = read_csv(paste0(valid_dir, "_annotations.csv"))

image_gen = image_data_generator(
  rescale = 1/255
)

train_generator = flow_images_from_dataframe(
  generator = image_gen,
  dataframe = train_annotations,
  directory = train_dir,
  target_size = c(target_image_height, target_image_width),
  batch_size = batch_size,
  class_mode = "categorical",
  color_mode = "grayscale",
  x_col = "filename",
  y_col = "class"
)

test_generator = flow_images_from_dataframe(
  generator = image_gen,
  dataframe = test_annotations,
  directory = test_dir,
  target_size = c(target_image_height, target_image_width),
  batch_size = batch_size,
  class_mode = "categorical",
  color_mode = "grayscale",
  x_col = "filename",
  y_col = "class"
)

valid_generator = flow_images_from_dataframe(
  generator = image_gen,
  dataframe = valid_annotations,
  directory = valid_dir,
  target_size = c(target_image_height, target_image_width),
  batch_size = batch_size,
  class_mode = "categorical",
  color_mode = "grayscale",
  x_col = "filename",
  y_col = "class"
)

output_dim = length(unique(train_generator$class_indices))

model = keras_model_sequential() %>%
  layer_conv_2d(
    kernel_size = c(3, 3),
    filters = 32,
    activation = 'relu',
    input_shape = c(target_image_height, target_image_width, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(
    kernel_size = c(3, 3),
    filters = 64,
    activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(
    kernel_size = c(3, 3),
    filters = 128,
    activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
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

predictions = predict(model, test_generator)

print(evaluation['accuracy'])

colnames(predictions) = names(train_generator$class_indices)

test_labels = select(test_annotations, filename, class)

final = bind_cols(as_tibble(predictions), test_labels)

print(final %>% slice_head(n = 6))

library(tidyverse)
suppressPackageStartupMessages(library(tidymodels))
suppressPackageStartupMessages(library(keras))

# training_data = read_csv("Agr_Food_CO2_training.csv")
# test_data = read_csv("Agr_Food_CO2_test.csv")
# 
# recp = recipe(`Average Temperature Celsius` ~ ., data = training_data) %>%
#   step_impute_median(all_numeric_predictors()) %>%
#   step_range(all_numeric_predictors(), min = 0, max = 1, clipping = FALSE) %>%
#   step_dummy(Area, one_hot = TRUE) %>%
#   prep(training = training_data)
# 
# training_data_baked = bake(recp, new_data = NULL)
# test_data_baked = bake(recp, new_data = test_data)
# 
# X_train = training_data_baked %>%
#   select(-`Average Temperature Celsius`) %>%
#   as.matrix()
# 
# y_train = pull(training_data_baked, `Average Temperature Celsius`)
# 
# X_test = test_data_baked %>%
#   select(-`Average Temperature Celsius`) %>%
#   as.matrix()
# 
# y_test = pull(test_data_baked, `Average Temperature Celsius`)
# 
# tensorflow::set_random_seed(33)
# 
# input_dim = ncol(X_train)
# output_dim = 1
# 
# model = keras_model_sequential() %>%
#   layer_dense(units = 64,
#               activation = "relu",
#               input_shape = input_dim) %>%
#   layer_dense(units = 64,
#               activation = "relu") %>%
#   layer_dense(units = output_dim)
# 
# compile(model, optimizer = "rmsprop",
#   loss = "mean_squared_error")
# 
# early_stop = callback_early_stopping(
#   monitor = "val_loss",
#   patience = 10,
#   restore_best_weights = TRUE
# )
# 
# history = model %>% fit(
#   x = X_train,
#   y = y_train,
#   epochs = 100,
#   batch_size = 32,
#   validation_split = 0.2,
#   callbacks = early_stop
# )
# 
# predictions = predict(model, X_test)
# 
# actual_pred = test_data %>%
#   select(Year, `Average Temperature Celsius`) %>%
#   bind_cols(tibble(predictions = as.vector(predictions)))
# 
# mtr = actual_pred %>%
#   metrics(truth = `Average Temperature Celsius`, estimate = predictions)
# 
# print(mtr)



# diff_activation_func = function(random_seed, activation_function) {
#   
#   training_data = read_csv("Agr_Food_CO2_training.csv")
#   test_data = read_csv("Agr_Food_CO2_test.csv")
#   
#   recp = recipe(`Average Temperature Celsius` ~ ., data = training_data) %>%
#     step_impute_median(all_numeric_predictors()) %>%
#     step_range(all_numeric_predictors(), min = 0, max = 1, clipping = FALSE) %>%
#     step_dummy(Area, one_hot = TRUE) %>%
#     prep(training = training_data)
#   
#   training_data_baked = bake(recp, new_data = NULL)
#   test_data_baked = bake(recp, new_data = test_data)
#   
#   X_train = training_data_baked %>%
#     select(-`Average Temperature Celsius`) %>%
#     as.matrix()
#   
#   y_train = pull(training_data_baked, `Average Temperature Celsius`)
#   
#   X_test = test_data_baked %>%
#     select(-`Average Temperature Celsius`) %>%
#     as.matrix()
#   
#   y_test = pull(test_data_baked, `Average Temperature Celsius`)
#   
#   tensorflow::set_random_seed(random_seed)
#   
#   input_dim = ncol(X_train)
#   output_dim = 1
#   
#   model = keras_model_sequential() %>%
#     layer_dense(units = 64,
#                 activation = activation_function,
#                 input_shape = input_dim) %>%
#     layer_dense(units = 64,
#                 activation = activation_function) %>%
#     layer_dense(units = output_dim)
#   
#   compile(model, optimizer = "rmsprop",
#           loss = "mean_squared_error")
#   
#   early_stop = callback_early_stopping(
#     monitor = "val_loss",
#     patience = 10,
#     restore_best_weights = TRUE
#   )
#   
#   history = model %>% fit(
#     x = X_train,
#     y = y_train,
#     epochs = 100,
#     batch_size = 32,
#     validation_split = 0.2,
#     callbacks = early_stop
#   )
#   
#   predictions = predict(model, X_test)
#   
#   actual_pred = test_data %>%
#     select(Year, `Average Temperature Celsius`) %>%
#     bind_cols(tibble(predictions = as.vector(predictions)))
#   
#   mtr = actual_pred %>%
#     metrics(truth = `Average Temperature Celsius`, estimate = predictions)
#   
#   rmse_value <- mtr %>%
#     filter(.metric == "rmse") %>%
#     pull(.estimate)
#   
#   return(rmse_value)
# }
# 
# activation_functions <- c("linear", "relu", "leaky_relu", "elu", "softplus", "swish")
# 
# random_seeds <- c(1, 2, 3)
# 
# results <- tibble(
#   activation_function = character(),
#   random_seed = integer(),
#   rmse = double()
# )
# 
# for (activation_func in activation_functions) {
#   for (seed in random_seeds) {
#     rmse_value <- diff_activation_func(random_seed = seed, activation_function = activation_func)
#     
#     results <- results %>% 
#       add_row(activation_function = activation_func, random_seed = seed, rmse = rmse_value)
#   }
# }
# 
# average_results <- results %>%
#   group_by(activation_function) %>%
#   summarize(avg_rmse = mean(rmse, na.rm = TRUE))
# 
# print(average_results)


TrackWeightsCallback = R6::R6Class("TrackWeightsCallback",
 inherit = KerasCallback,
 
 public = list(
   weights_history = NULL,
   
   on_train_begin = function(logs = NULL) {
     self$weights_history <- list()  # Initialize empty list
   },
   
   on_epoch_end = function(epoch, logs = NULL) {
     current_weights <- lapply(self$model$layers, function(layer) layer$get_weights())
     self$weights_history[[epoch + 1]] <- current_weights
   }
 )
)


training_data = read_csv("Agr_Food_CO2_training.csv")
test_data = read_csv("Agr_Food_CO2_test.csv")

recp = recipe(`Average Temperature Celsius` ~ ., data = training_data) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1, clipping = FALSE) %>%
  step_dummy(Area, one_hot = TRUE) %>%
  prep(training = training_data)

training_data_baked = bake(recp, new_data = NULL)
test_data_baked = bake(recp, new_data = test_data)

X_train = training_data_baked %>%
  select(-`Average Temperature Celsius`) %>%
  as.matrix()

y_train = pull(training_data_baked, `Average Temperature Celsius`)

X_test = test_data_baked %>%
  select(-`Average Temperature Celsius`) %>%
  as.matrix()

y_test = pull(test_data_baked, `Average Temperature Celsius`)

tensorflow::set_random_seed(33)

input_dim = ncol(X_train)
output_dim = 1

model = keras_model_sequential() %>%
  layer_dense(units = 4,
              activation = "relu",
              input_shape = input_dim) %>%
  layer_dense(units = 4,
              activation = "relu") %>%
  layer_dense(units = output_dim)

compile(model, optimizer = "rmsprop",
        loss = "mean_squared_error")

early_stop = callback_early_stopping(
  monitor = "val_loss",
  patience = 10,
  restore_best_weights = TRUE
)

track_weights_cb = TrackWeightsCallback$new()

history = model %>% fit(
  x = X_train,
  y = y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2,
  callbacks = list(track_weights_cb)
)

predictions = predict(model, X_test)

actual_pred = test_data %>%
  select(Year, `Average Temperature Celsius`) %>%
  bind_cols(tibble(predictions = as.vector(predictions)))

mtr = actual_pred %>%
  metrics(truth = `Average Temperature Celsius`, estimate = predictions)

weights_history = track_weights_cb$weights_history

epoch_weights <- matrix(nrow = length(weights_history), ncol = 4)

for (i in 1:length(weights_history)) {
  current_epoch_weights = weights_history[[i]][[1]][[1]][1:4, 1]
  
  epoch_weights[i, ] <- current_epoch_weights
}

print(epoch_weights)






# weights = model %>%
#   get_layer(index = 1) %>%
#   get_weights()
# 
# weights = weights[[1]]
# 
# print(weights[1:4, 1])


# mod = rand_forest(trees = 10, mode = "regression") %>%
#   set_engine("ranger") %>%
#   fit(`Average Temperature Celsius` ~ ., data = training_data_baked)
# 
# predictions = predict(mod, new_data = test_data_baked)
# 
# actual_pred = select(test_data, Year, `Average Temperature Celsius`) %>%
#   cbind(predictions) %>%
#   as.tibble() %>%
#   dplyr::rename(predictions = .pred) %>%
#   metrics(truth = `Average Temperature Celsius`, estimate = predictions) %>%
#   print()
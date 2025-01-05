library(tidyverse)
library(tidymodels)
library(keras)

data = read_tsv("GDSC.tsv.gz")

filtered_data = select(data, -Cell_Line_ID, -Cell_Line_Name, -Tissue_Type)

tensorflow::set_random_seed(0)

data_split = initial_split(filtered_data, prop = 0.75)

train_data = training(data_split)
test_data = testing(data_split)

recp = recipe(Gefitinib_IC50 ~., data = train_data) %>%
  step_normalize(all_numeric_predictors()) %>%
  prep(training = train_data)

training_data_baked = bake(recp, new_data = NULL)
test_data_baked = bake(recp, new_data = test_data)

X_train = training_data_baked %>%
  select(-Gefitinib_IC50) %>%
  as.matrix()

y_train = pull(training_data_baked, Gefitinib_IC50)

X_test = test_data_baked %>%
  select(-Gefitinib_IC50) %>%
  as.matrix()

y_test = pull(test_data_baked, Gefitinib_IC50)

input_dim = ncol(X_train)
output_dim = 1

model = keras_model_sequential() %>%
  layer_dense(units = 64,
              activation = "relu",
              input_shape = input_dim) %>%
  layer_dense(units = 64,
              activation = "relu") %>%
  layer_dense(units = output_dim)

compile(model, optimizer = "rmsprop",
  loss = "mean_squared_error")

early_stop = callback_early_stopping(
  monitor = "val_loss",
  patience = 10,
  restore_best_weights = TRUE
)

history = model %>% fit(
  x = X_train,
  y = y_train,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2,
  callbacks = early_stop
)

predictions = predict(model, X_test)

actual_pred = test_data %>%
  select(Gefitinib_IC50) %>%
  bind_cols(tibble(predictions = as.vector(predictions)))

mtr = actual_pred %>%
  metrics(truth = Gefitinib_IC50, estimate = predictions)

print(mtr)

#############################################################

tensorflow::set_random_seed(0)

model = keras_model_sequential() %>%
  layer_dense(units = 64,
              activation = "relu",
              input_shape = input_dim) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 64,
              activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = output_dim)

compile(model, optimizer = "rmsprop",
        loss = "mean_squared_error")

early_stop = callback_early_stopping(
  monitor = "val_loss",
  patience = 10,
  restore_best_weights = TRUE
)

history = model %>% fit(
  x = X_train,
  y = y_train,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2,
  callbacks = early_stop
)

predictions = predict(model, X_test)

actual_pred = test_data %>%
  select(Gefitinib_IC50) %>%
  bind_cols(tibble(predictions = as.vector(predictions)))

mtr = actual_pred %>%
  metrics(truth = Gefitinib_IC50, estimate = predictions)

print(mtr)

#############################################################

tensorflow::set_random_seed(0)

model = keras_model_sequential() %>%
  layer_dense(units = 64,
              activation = "relu",
              input_shape = input_dim) %>%
  layer_dropout(rate = 0.5) %>%
  layer_batch_normalization() %>%
  layer_dense(units = 64,
              activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_batch_normalization() %>%
  layer_dense(units = output_dim)

compile(model, optimizer = "rmsprop",
        loss = "mean_squared_error")

early_stop = callback_early_stopping(
  monitor = "val_loss",
  patience = 10,
  restore_best_weights = TRUE
)

history = model %>% fit(
  x = X_train,
  y = y_train,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2,
  callbacks = early_stop
)

predictions = predict(model, X_test)

actual_pred = test_data %>%
  select(Gefitinib_IC50) %>%
  bind_cols(tibble(predictions = as.vector(predictions)))

mtr = actual_pred %>%
  metrics(truth = Gefitinib_IC50, estimate = predictions)

print(mtr)

#############################################################

diff_optimizer_func = function(random_seed, optimizer) {
  
  tensorflow::set_random_seed(random_seed)
  
  model = keras_model_sequential() %>%
    layer_dense(units = 64,
                activation = "relu",
                input_shape = input_dim) %>%
    layer_dropout(rate = 0.5) %>%
    layer_batch_normalization() %>%
    layer_dense(units = 64,
                activation = "relu") %>%
    layer_dropout(rate = 0.3) %>%
    layer_batch_normalization() %>%
    layer_dense(units = output_dim)
  
  compile(model, optimizer = optimizer,
          loss = "mean_squared_error")
  
  early_stop = callback_early_stopping(
    monitor = "val_loss",
    patience = 10,
    restore_best_weights = TRUE
  )
  
  history = model %>% fit(
    x = X_train,
    y = y_train,
    epochs = 100,
    batch_size = 32,
    validation_split = 0.2,
    callbacks = early_stop
  )
  
  predictions = predict(model, X_test)
  
  actual_pred = test_data %>%
    select(Gefitinib_IC50) %>%
    bind_cols(tibble(predictions = as.vector(predictions)))
  
  mtr = actual_pred %>%
    metrics(truth = Gefitinib_IC50, estimate = predictions)
  
  rmse_value <- mtr %>%
    filter(.metric == "rmse") %>%
    pull(.estimate)

  return(rmse_value)
}

optimizers <- c("rmsprop", "sgd", "adagrad", "adadelta", "adam", "adamax", "ftrl", "nadam")

random_seeds <- c(1, 2, 3)

results <- tibble(
  Optimizer = character(),
  rmse = double()
)

for (optimizer in optimizers) {
  for (seed in random_seeds) {
    rmse_value <- diff_optimizer_func(random_seed = seed, optimizer = optimizer)

    results <- results %>%
      add_row(Optimizer = optimizer, rmse = rmse_value)
  }
}

average_results <- results %>%
  group_by(Optimizer) %>%
  summarize(mean_rmse = mean(rmse)) %>%
  arrange(mean_rmse)

print(average_results)



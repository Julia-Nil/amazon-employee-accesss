# Load packages
library(vroom)
library(lubridate)
library(dplyr)
library(tidyr)
library(ggplot2)
library(tidymodels)
library(recipes)
library(embed)
library(doParallel)

# --- Parallel setup ---
num_cores <- parallel::detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# --- Load data ---
amazontrain <- vroom("train.csv")
amazontest  <- vroom("test.csv")

# Convert outcome to factor
amazontrain$ACTION <- as.factor(amazontrain$ACTION)

# --- Recipe with PCA ---
amazonrecipetrain <- recipe(ACTION ~ ., data = amazontrain) %>%
  step_other(all_nominal_predictors(), threshold = 0.1) %>% 
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_numeric_predictors(), threshold = 0.95)

# Bake once to know number of predictors
baked_train <- bake(prep(amazonrecipetrain), amazontrain)
n_pcs <- ncol(baked_train) - 1  # subtract outcome

# --- Random Forest workflow ---
amazon_log_model <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 500
) %>%
  set_engine("ranger") %>%
  set_mode("classification")

amazon_wf <- workflow() %>%
  add_recipe(amazonrecipetrain) %>%
  add_model(amazon_log_model)

# --- Cross-validation ---
folds <- vfold_cv(amazontrain, v = 5)

# --- Tune parameters safely ---
tune_params <- extract_parameter_set_dials(amazon_wf) %>%
  update(mtry = mtry(range = c(1, n_pcs))) %>% 
  finalize(amazontrain)

tuning_grid <- grid_regular(tune_params, levels = 5)

# --- Run tuning ---
CV_results <- tune_grid(
  amazon_wf,
  resamples = folds,
  grid = tuning_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(save_pred = TRUE, verbose = TRUE)
)

# --- Select best params & fit final model ---
best_params <- select_best(CV_results, metric = "roc_auc")

final_wf <- amazon_wf %>% 
  finalize_workflow(best_params) %>% 
  fit(data = amazontrain)

# --- Predict on test set ---
preds <- predict(final_wf, new_data = amazontest, type = "prob") %>%
  rename(ACTION = .pred_1)

# --- Save results ---
results <- tibble(id = amazontest$id, ACTION = preds$ACTION)
vroom_write(results, "rfbatchresults.csv", delim = ",")

# --- Stop parallel cluster ---
stopCluster(cl)

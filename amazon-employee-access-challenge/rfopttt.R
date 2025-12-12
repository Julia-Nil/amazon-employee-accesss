
library(vroom)
library(lubridate)
library(tidymodels)
library(recipes)
library(embed)
library(dials)
library(parallel)
library(doParallel)
library(tune)


# --- Parallel setup ---
num_cores <- parallel::detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# --- Load data ---
amazontrain <- vroom("train.csv") %>%
  mutate(ACTION = factor(ACTION))

amazontest  <- vroom("test.csv")


amazonrecipetrain <- recipe(ACTION ~ ., data = amazontrain) %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = "ACTION") %>%
  step_normalize(all_numeric_predictors())






amazon_log_model <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 100
) %>%
  set_engine("ranger") %>%
  set_mode("classification")



amazon_wf <- workflow() %>%
  add_recipe(amazonrecipetrain) %>%
  add_model(amazon_log_model)




folds <- vfold_cv(amazontrain, v = 5, repeats =1) # higher v is maybe better but slow 10ish

tune_params <- extract_parameter_set_dials(amazon_wf)
tune_params <- tune_params %>%
  finalize(amazontrain)


tuning_grid <- grid_regular(tune_params, levels = 5)

CV_results <- tune_grid(
  amazon_wf,
  resamples = folds,
  grid = tuning_grid,
  metrics = metric_set(roc_auc)
)



best_params <- select_best(CV_results, metric="roc_auc")

final_wf <- amazon_wf %>% 
  finalize_workflow(best_params) %>% 
  fit(data=amazontrain)




preds <- predict(final_wf, new_data=amazontest, type = "prob") %>% 
  rename(ACTION=.pred_1)


# --- Save results ---
results <- tibble(id = amazontest$id, ACTION = preds$ACTION)
vroom_write(results, "rfopttresults.csv", delim = ",")

# --- Stop parallel cluster ---
stopCluster(cl)


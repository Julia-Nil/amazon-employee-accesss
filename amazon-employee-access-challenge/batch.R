
  
library(vroom)
library(lubridate)
library(dplyr)
library(ggplot2)
library(readr)
library(purrr)
library(tidyr)
library(tidymodels)
library(recipes)
library(embed)
library(doParallel)

# Detect and register available cores
num_cores <- parallel::detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)

amazontrain <- vroom("train.csv")

amazontest <- vroom("test.csv")



id_cols <- c("RESOURCE","MGR_ID","ROLE_ROLLUP_1","ROLE_ROLLUP_2",
             "ROLE_DEPTNAME","ROLE_TITLE","ROLE_FAMILY_DESC",
             "ROLE_FAMILY","ROLE_CODE")

amazontrain <- amazontrain %>% mutate(across(all_of(id_cols), as.factor))
amazontest  <- amazontest  %>% mutate(across(all_of(id_cols), as.factor))

amazontrain <- amazontrain %>%
  mutate_if(is.character, as.factor) %>%   # convert text to factors
  mutate_if(is.numeric, as.factor)         # convert IDs to factors if they’re really categories

amazonrecipetrain <- recipe(ACTION ~ ., data = amazontrain) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>% 
  step_unknown(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors())
#step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))


amazon_log_model <- logistic_reg(mixture=tune(), penalty = tune()) %>% 
  set_engine("glmnet")



amazon_wf <- workflow() %>% 
  add_recipe(amazonrecipetrain) %>% 
  add_model(amazon_log_model)



folds <- vfold_cv(amazontrain, v=20)
tuning_grid <- grid_regular(penalty(), mixture(), levels=10)

CV_results <- tune_grid(
  amazon_wf,
  resamples = folds,
  grid = tuning_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(save_pred = TRUE, verbose = TRUE)
)

stopCluster(cl)
registerDoSEQ()


best_params <- select_best(CV_results, metric="roc_auc")

final_wf <- amazon_wf %>% 
  finalize_workflow(best_params) %>% 
  fit(data=amazontrain)




preds <- predict(final_wf, new_data=amazontest, type = "prob") %>% 
  rename(ACTION=.pred_1)



results <- tibble(id = amazontest$id, ACTION = preds$ACTION)

vroom_write(results, "batchresultsbeefy.csv", delim = ",")

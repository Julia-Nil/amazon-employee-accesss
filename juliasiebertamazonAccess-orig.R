library(tidyverse)
library(tidymodels)
library(embed)
library(ggmosaic)
library(vroom)

amazon_train <- vroom("train.csv") %>% 
  mutate(ACTION = factor(ACTION))
amazon_test <- vroom("test.csv") 

##########
### EDA
##########
amazon_train %>% 
  mutate(ROLE_FAMILY = factor(ROLE_FAMILY)) %>% 
  group_by(ROLE_FAMILY) %>% 
  summarize(n = n()) %>% 
  arrange(desc(n)) %>% 
  head(n = 10) %>% 
ggplot() +
  geom_col(aes(x = ROLE_FAMILY, y = n)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

amazon_train %>% 
  mutate(ROLE_DEPTNAME = factor(ROLE_DEPTNAME)) %>% 
  group_by(ROLE_DEPTNAME) %>% 
  summarize(n = n()) %>% 
  arrange(desc(n)) %>% 
  head(n = 10) %>% 
ggplot() +
  geom_col(aes(x = ROLE_DEPTNAME, y = n)) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

###########
### Recipe
##########
# amazon_recipe <- recipe(ACTION ~., data=amazon_train) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   step_other(all_factor_predictors(), threshold = .001) %>%
#   #step_lencode_mixed(all_factor_predictors(), outcome = vars(ACTION)) %>% Target Encoding
#   step_dummy(all_factor_predictors())  %>% 
#   step_normalize(all_factor_predictors()) %>% 
#   step_pca(all_predictors(), threshold = 0.8)

#####
## SVM Recipe
#####
# amazon_recipe <- recipe(ACTION ~ ., data = amazon_train) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
#   step_normalize(all_numeric_predictors()) %>%
#   step_zv(all_predictors()) # remove zero-variance cols

#####
## Balancing Data Recipe
#####
amazon_recipe <- recipe(ACTION ~ ., data = amazon_train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors())

prep <- prep(amazon_recipe)
baked <- bake(prep, new_data = amazon_train)
##########
## Models
##########

#####
## logistic regression model
#####
# logRegModel <- logistic_reg() %>% 
#   set_engine("glm")

#####
## penalized regression model
#####
# penLogModel <- logistic_reg(mixture = tune(), penalty = tune()) %>%
#   set_engine("glmnet")

#####
## random forest
#####
forest_mod <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees = 100) %>%
  set_engine("ranger") %>%
  set_mode("classification")

#####
## knn model
#####
# knn_model <- nearest_neighbor(neighbors = tune()) %>%
#   set_mode("classification") %>%
#   set_engine("kknn")

#####
## Naive bayes model
#####
# nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
#   set_mode("classification") %>%
#   set_engine("naivebayes")

#####
## SVM models
#####
# svm_Radial <- svm_rbf(rbf_sigma = 0.177, cost = 0.00316) %>%
#   set_mode("classification") %>%
#   set_engine("kernlab")
# 
# svm_Poly <- svm_poly(degree = 1, cost = 0.0131) %>%
#   set_mode("classification") %>%
#   set_engine("kernlab")
# 
# svm_Linear <- svm_linear(cost = 0.0131) %>%
#   set_mode("classification") %>%
#   set_engine("kernlab")



##########
## Put into a workflow here
##########

#####
## logistic regression
#####
# log_reg_workflow <- workflow() %>%
#   add_recipe(amazon_recipe) %>%
#   add_model(logRegModel) %>%
#   fit(data = amazon_train)

#####
## penalized logistic regression
#####
# pen_workflow <- workflow() %>%
#   add_recipe(amazon_recipe) %>%
#   add_model(penLogModel)

#####
## random forest
#####
forest_workflow <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(forest_mod)

#####
## knn 
#####
# knn_wf <- workflow() %>%
#   add_recipe(amazon_recipe) %>%
#   add_model(knn_model)

#####
## Naive Bayes workflow
#####
# nb_wf <- workflow() %>%
#   add_recipe(amazon_recipe) %>%
#   add_model(nb_model)

#####
## SVM workflows
#####
# poly_wf <- workflow() %>% 
#   add_model(svm_Poly) %>% 
#   add_recipe(amazon_recipe) %>% 
#   fit(data = amazon_train)
# 
# radial_wf <- workflow() %>%
#   add_model(svm_Radial) %>%
#   add_recipe(amazon_recipe) %>%
#   fit(data = amazon_train)
# 
# linear_wf <- workflow() %>%
#   add_model(svm_Linear) %>% 
#   add_recipe(amazon_recipe) %>%
#   fit(data = amazon_train)


##########
## CV
##########

#####
## Grid of values to tune over
#####
# tuning_grid <- grid_regular(penalty(),
#                             mixture(),
#                             levels = 4)

#####
## Grid for forest
#####
tuning_grid <- grid_regular(mtry(range = c(1, 9)),
                            min_n(), 
                            levels = 5)

#####
## Grid for knn
#####
# tuning_grid <- grid_regular(neighbors(),
#                             levels = 5)

#####
## Grid for naive bayes
#####
# tuning_grid <- grid_regular(Laplace(),
#                             smoothness(),
#                             levels = 4)

#####
## Grid for SVM
#####
# tuning_grid_poly <- grid_regular(degree(), 
#                             cost(), 
#                             levels = 5)
# 
# tuning_grid_radial <- grid_regular(rbf_sigma(), 
#                             cost(), 
#                             levels = 5)
# 
# tuning_grid_linear <- grid_regular(cost(), 
#                             levels = 5)

#####
## Split data for CV
#####
folds <- vfold_cv(amazon_train, v = 5, repeats = 1)

#####
## Run CV
#####
CV_results <- tune_grid(
    forest_workflow,
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(roc_auc))

#####
## Find best tuning parameters
#####
bestTune <- CV_results %>%
  select_best(metric="roc_auc")


##########
## Predictions
##########

#####
## Finalize workflow and fit it
#####
final_wf <- forest_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = amazon_train)

#####
## Make predictions
#####
amazon_predictions <- predict(final_wf,
                              new_data=amazon_test,
                              type="prob") # "class" or "prob"

#####
## Format predictions
#####
kaggle_predictions <- amazon_predictions %>% 
  bind_cols(., amazon_test) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION = .pred_1, 
         Id = id)

vroom_write(x = kaggle_predictions, file = "./RandomForest.csv", delim = ",")

## Create tuning graphic
# CV_results %>% collect_metrics() %>% 
#   filter(.metric=="roc_auc") %>% 
#   ggplot(aes(x = hidden_units, y = mean)) +
#   geom_line()

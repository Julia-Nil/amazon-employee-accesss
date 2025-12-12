## Read in the Data
train <- vroom("C:/Users/julia/OneDrive - Brigham Young University/Desktop/stat 348/amazon-employee-access/amazon-employee-access-challenge/train.csv") %>%
  mutate(ACTION=factor(ACTION))
test <- vroom("C:/Users/julia/OneDrive - Brigham Young University/Desktop/stat 348/amazon-employee-access/amazon-employee-access-challenge/test.csv")

## Define the folds up front
folds <- vfold_cv(train, v=10)
metSet <- metric_set(roc_auc)

## Define a Recipe
amazonRecipe <- recipe(ACTION~., data=train) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.001) %>%
  step_dummy(all_nominal_predictors()) 

## Define the model
logReg_model <- logistic_reg() %>%
  set_engine("glm")

## Set up workflow
system.time({logReg_wf <- workflow() %>%
  add_recipe(amazonRecipe) %>%
  add_model(logReg_model) %>%
  fit(data=train)})

## Predict test set
logRegPreds <- logReg_wf %>%
  predict(new_data=test, type="prob") %>%
  bind_cols(test) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)

## Write out the Predictions
vroom_write(x=logRegPreds, file="./LogRegPreds.csv", delim=",")
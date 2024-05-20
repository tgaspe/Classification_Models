# Stats Project
## Overview
This project, titled Stats Project 2, was conducted by Theodoro Gasperin Terra Camargo and Tarek Chammaa El Rifai Mashmoushi as part of the Statistical Methods for Bioinformatics class. The project's primary goal is to analyze a dataset with 102 columns (101 predictor variables and 1 response variable) using various statistical and machine learning models to make predictions, particularly focusing on model performance in terms of accuracy, specificity, and sensitivity.

## Project Structure
### Setup
r
Copy code
# Load necessary packages
library(mice)
library(caret)
library(glmnet)
library(dplyr)
library(gam)
library(randomForest)
library(pROC)

# Import dataset
load("./MI.RData")

# Function to rotate a Confusion Matrix by 180 degrees
rotate_table_180 <- function(tbl) {
  if (!is.matrix(tbl)) {
    stop("Input must be a matrix")
  }
  rotated_tbl <- tbl[nrow(tbl):1, ncol(tbl):1]
  return(rotated_tbl)
}
Data Exploration and Imputation
Data Exploration:

Dimensionality: The dataset contains 102 columns, including 101 predictors and 1 response variable (LET_IS), which is binary.
Missing Values: Identified columns with missing values and rows with missing data.
Imputation:

Used the mice package for predictive imputation to handle missing data, creating a complete dataset for analysis.
r
Copy code
# Check dimensions and missing values
dim(MI)
sum(is.na(MI))

# Impute missing values
MI_imputed <- mice(MI, m=1, maxit=50, method="pmm", seed=500)
MI <- complete(MI_imputed)
Data Splitting
Split the dataset into training (80%) and testing (20%) sets.

r
Copy code
set.seed(33)
trainIndex <- createDataPartition(MI$LET_IS, p = .8, list = FALSE, times = 1)
MI_train <- MI[trainIndex, ]
MI_test <- MI[-trainIndex, ]
Modeling and Analysis
Unconstrained Linear Model:

Fitted a logistic regression model to the training data and evaluated its performance.
Lasso and Ridge Regression:

Applied Lasso (L1 regularization) and Ridge (L2 regularization) regression models using cross-validation to identify the best lambda values.
Compared models based on accuracy, specificity, and sensitivity.
r
Copy code
# Lasso and Ridge Regression
lasso_model <- cv.glmnet(as.matrix(select(MI_train, -LET_IS)), MI_train$LET_IS, type.measure = "class", alpha = 1, family = "binomial", nfolds = 5)
ridge_model <- cv.glmnet(as.matrix(select(MI_train, -LET_IS)), MI_train$LET_IS, type.measure = "class", alpha = 0, family = "binomial", nfolds = 5)

# Predictions and performance evaluation
lasso_pred.min <- predict(lasso_model, s= lasso_model$lambda.min, newx = as.matrix(select(MI_test, -LET_IS)), type = "class")
ridge_pred.min <- predict(ridge_model, s= ridge_model$lambda.min, newx = as.matrix(select(MI_test, -LET_IS)), type = "class")
Generalized Additive Model (GAM):

Assessed non-linear effects among top predictors.
Fitted a GAM and compared its performance with the linear models.
Random Forest:

Fitted a Random Forest model to capture complex interactions and non-linear relationships.
Compared the performance of the Random Forest model with linear models and GAM.
r
Copy code
# Random Forest model
forest_model <- randomForest(LET_IS ~ ., mtry= 18, data = MI_train, ntree = 500)
forest_pred <- predict(forest_model, MI_test, type = "class")
ROC Curve Comparison:
Generated ROC curves for each model to compare their performance using the AUC metric.
r
__Copy code__
# ROC Curve comparison
roc.forest_pred <- roc(MI_test$LET_IS, as.numeric(forest_pred))
roc.lasso_pred <- roc(MI_test$LET_IS, as.numeric(as.factor(lasso_pred.min)))
roc.ridge_pred <- roc(MI_test$LET_IS, as.numeric(as.factor(ridge_pred.min)))
roc.gam_pred <- roc(MI_test$LET_IS, as.numeric(gam_pred_binary))
roc.unconstrained_pred_class <- roc(MI_test$LET_IS, as.numeric(unconstrained_pred_class))

plot(roc.unconstrained_pred_class, col = "blue", main = "Comparison of ROC Curves", print.auc = TRUE)
plot(roc.forest_pred, col = "red", add = TRUE, print.auc = TRUE)
plot(roc.gam_pred, col = "green", add = TRUE, print.auc = TRUE)
plot(roc.lasso_pred, col = "purple", add = TRUE, print.auc = TRUE)
plot(roc.ridge_pred, col = "orange", add = TRUE, print.auc = TRUE)
legend("bottomright", legend = c("Unconstrained", "Random Forest", "GAM", "Lasso", "Ridge"), col = c("blue", "red", "green", "purple", "orange"), lwd = 2)

## Conclusions
Lasso Regression showed improved sensitivity and overall performance, making it the most suitable model for our goals.
GAM was effective in capturing non-linear effects but did not outperform Lasso regression in all metrics.
Random Forest demonstrated high sensitivity but did not capture complex relationships as effectively as expected compared to other models.
This project highlights the importance of selecting appropriate models based on the specific objectives and characteristics of the dataset, such as high dimensionality, non-linear effects, and the presence of missing data.

## How to Run
Load the dataset by placing MI.RData in the project directory.
Run the R script provided to perform data exploration, imputation, and model fitting.
Evaluate model performance and compare results using the provided ROC curve plots and confusion matrices.
Dependencies
R (version 4.0 or higher)
Packages: mice, caret, glmnet, dplyr, gam, randomForest, pROC
Contact
For any questions or issues, please contact Theodoro Gasperin Terra Camargo and Tarek Chammaa El Rifai Mashmoushi.

__Date:__ 2024-05-13
__Class:__ Statistical Methods for Bioinformatics

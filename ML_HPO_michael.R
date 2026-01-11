#import relevant libraries

install.packages("parallelly")
library(mlr3)
library(tidyverse)
library(mlr3verse)
library(ranger)
library(mlr3learners)

#set working directory
setwd("C:/Users/mutah/OneDrive/Desktop/data project/RAS_Cleaned_data")

# read the file
data <- read_csv("cleaned_merge_data.csv")

#data dimensions
dim(data)

#first 6 observations
head(data)

#structure of the data
str(data)

#summary of the data
summary(data)


# converting out data to data.table
data_new <- as.data.table(data)  
data_new %>% view()#
data_new


# =======================================================================================
# STEP 1: FEATURE ENGINEERING
# =======================================================================================
# load relevant libraries

library(data.table)

data_df <- data_new %>%
  mutate(
    # 1. Equipment age in days
    equipment_age_days = as.numeric(traindate - applieddate),
    
    # 2. Previous failures history
    prev_failures = shift(cumsum(failedin30days), 1, fill = 0),
    
    # 3. Cumulative mileage
    cumulative_mileage = partmileage + addedmileage,
    
    # 4. Total travel miles
    total_travel_miles = loadedtravelledmiles + emptytravelledmiles,
    
    # 5. Utilization ratio
    utilization_ratio = ifelse((loadedtravelledmiles + emptytravelledmiles) == 0, NA,
                               loadedtravelledmiles / (loadedtravelledmiles + emptytravelledmiles)),
    
    # 6. Travel efficiency
    travel_efficiency = ifelse(equipment_age_days == 0, NA, 
                               (loadedtravelledmiles + emptytravelledmiles) / equipment_age_days),
    
    # 7. Average speed
    average_speed = ifelse(equipment_age_days == 0, NA, 
                           (loadedtravelledmiles + emptytravelledmiles) / equipment_age_days),
    
    # 8. Dynamic loading score (weighted or summed - here, simple average)
    dynamic_loading_score = rowMeans(select(., dynamicvertical, dynamicratio, huntingindex, 
                                            leftrightimbalance, percentload), na.rm = TRUE),
    
    # 9. Shock stress
    shock_stress = maxvertical - averagevertical,
    
    # 10. Stress ratio
    stress_ratio = ifelse(averagevertical == 0, NA, dynamicvertical / averagevertical),
    
    # 11. Applied year and month
    applied_year = year(applieddate),
    applied_month = month(applieddate),
    
    # 12. Applied age in days relative to record month
    applied_age_days = as.integer(as.Date(paste0(recordmonth, "-01")) - applieddate)
   
  )

data_df %>% dim
data_df %>% view()

# =========================================================================================
# STEP 2: DATA PREPARATION & BASIC CLEANING
# =========================================================================================

#converting character columns to factors for proper handling 
cols_to_factor <- c("recordmonth", "asbuilt", "truck", "side", "failurereason", "direction")

data_df[, (cols_to_factor) := lapply(.SD, as.factor), .SDcols = cols_to_factor]

# Convert target to factor for classification
data_df[, failurereason := as.factor(failurereason)]

# Clean factor levels (remove spaces)
#levels(data_df$failurereason) <- make.names(levels(data_df$failurereason))

# Handle missing values
data_df[is.na(failurereason), failurereason := "No_Failure"]

# Remove problematic columns
data_df[, "...1" := NULL]

data_df %>% view()
data_df %>% str

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# STEP 3: EXPLORATORY DATA ANALYSIS       
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# check for correlation
# Get only numeric columns

num_cols <- names(which(sapply(data_df, is.numeric)))
cor_matrix <- cor(data_df[, ..num_cols], use = "pairwise.complete.obs")

# View high correlations
library(corrplot)
corrplot(cor_matrix, method = "color", tl.cex = 0.7, type = "upper", order = "hclust")

# corrplot is unreadable due to very many features and would yield a very long impractical table
# therefore we pick top 20 features by variance and plot the correlation
top_vars <- names(sort(sapply(data_df[, ..num_cols], var), decreasing = TRUE))[1:20]
cor_matrix_small <- cor(data_df[, ..top_vars], use = "pairwise.complete.obs")

# Plot
library(corrplot)
corrplot(cor_matrix_small, method = "color", tl.cex = 0.8, order = "hclust")
#from our corrplot we see high correlation between site x and site y, average vertical and max vertical, travel miles and train speed

#re-oder columns
data_model <- data_df %>%
  relocate(failedin30days, failurereason, .after = last_col())
colnames(data_model)

#Visualize few common features for Outliers
# Convert to long format
plot_data <- melt(data_model, measure.vars = c("prev_failures", "huntingindex", "dynamic_loading_score", "dynamicratio"))

# Facet by variable (separate boxplots)
ggplot(plot_data, aes(x = variable, y = value, fill = variable)) +
  geom_boxplot(outlier.colour = "red", outlier.shape = 1) +
  facet_wrap(~ variable, scales = "free_y") +  # Different y-scales per plot
  scale_fill_manual(values = c(
    "prev_failures" = "skyblue",
    "huntingindex" = "blue",
    "dynamic_loading_score" = "lightgreen",
    "dynamicratio" = "salmon"
  )) +
  theme_minimal() +
  labs(title = "Boxplots of Key Features (Different Scales)", x = "Feature", y = "Value")

# Basic data inspection
data_model %>% dim() #dimension
data_model %>% str() # structure

#probability of positive outcome
print(table(data_model$failedin30days)) #shows probability of all failure reasons combined
cat("Class imbalance ratio:", round(sum(data_model$failedin30days == "1") / nrow(data_model) * 100, 2), "%\n\n")

# ========================================================================================
# STEP 4: SPLIT DATA
# ========================================================================================

# Set a random seed for reproducibility
set.seed(123)

# Create a row index for training (e.g. 80%)
train_idx <- sample(seq_len(nrow(data_model)), size = 0.8 * nrow(data_model))

# Create train and test data
train_data <- data_model[train_idx]
test_data <- data_model[-train_idx]

# ========================================================================================
# STEP 5: CREATE TASK
# ========================================================================================

# Drop date columns (after feature engineering)
train_data[, c("recordmonth", "applieddate", "traindate") := NULL]
test_data[, c("recordmonth", "applieddate", "traindate") := NULL]

# Create classification task
task <- TaskClassif$new(
  id = "wheel_failure_bench",
  backend = train_data,
  target = "failurereason"
)
#task$positive <- "1"  # Set positive class for metrics like AUC, recall, logloss, F1, precision

task %>% autoplot()
ggsave("task.pdf")
# =========================================================================================
# STEP 6: DEFINE RESAMPLING STRATEGY
# =========================================================================================

set.seed(431)
resampling <- rsmp("cv", folds = 5)
resampling$instantiate(task)

# =========================================================================================
# STEP 7: PIPELINE GRAPH(IMPUTATION + TREATMENT ENCODING + BALANCING)
# =========================================================================================

library(mlr3pipelines)

# Create preprocessing graph: handle factors + missing values
#check for missing values
sum(is.na(train_data))
sum(is.na(test_data))
#our data has only one missing data, we apply simple median imputation

graph_onehot <- 
  po("removeconstants") %>>%
  po("imputemedian") %>>%       # simple median imputation
  po("encode", method = "one-hot")

# Add class balancing
preprocess_graph <- po("classbalancing",
                       id = "balancer",
                       adjust = "major",
                       reference = "minor",
                       ratio = 4,
                       shuffle = TRUE) %>>%
  graph_onehot

# =========================================================================================
# STEP 8: LEARNERS WITH PREPROCESSING
# =========================================================================================
learners_bench <- list(
  # Baseline
  GraphLearner$new(
    preprocess_graph %>>% lrn("classif.featureless", predict_type = "prob"),
    id = "baseline"
  ),
  
  # Ranger
  GraphLearner$new(
    preprocess_graph %>>% 
      lrn("classif.ranger", predict_type = "prob",
          mtry.ratio = 0.5,
          min.node.size = 5,
          num.trees = 100),
    id = "ranger"
  ),
  
  # XGBoost
  GraphLearner$new(
    preprocess_graph %>>%
      lrn("classif.xgboost", predict_type = "prob",
          nrounds = 100, eta = 0.1,
          max_depth = 6,
          subsample = 0.8,
          colsample_bytree = 0.8,
          scale_pos_weight = 20),
    id = "xgboost"
  ),
  
  # Penalized Logistic Regression (Elastic Net)
  GraphLearner$new(
    preprocess_graph %>>%
      po("scale") %>>%
      lrn("classif.cv_glmnet", predict_type = "prob",
          alpha = 0.5, nfolds = 5),
    id = "logistic"
  ),
  
  # SVM
  GraphLearner$new(
    preprocess_graph %>>%
      po("scale") %>>%
      lrn("classif.svm", predict_type = "prob",
          cost = 1, gamma = 0.01,
          kernel = "radial",
          type = "C-classification"
          ),
    id = "svm"
  )
)

# ==========================================================================================
# STEP 9A: BENCHMARK EVALUATION
# ==========================================================================================

# Define appropriate performance metrics
measures_bench <- msr("classif.logloss")



# Create benchmark design
design_bench <- benchmark_grid(
  tasks = task,
  learners = learners_bench,
  resamplings = resampling
)

# Run the benchmark
bmr <- benchmark(design_bench)

# Aggregate and inspect results
results <- bmr$aggregate(measures_bench)
print(results)
#The Logistic regression and xgboost have the lowest log loss

bmr %>% autoplot()
bmr %>% 
  autoplot() +
  labs(
    x = "Learner",
    y = "Log Loss"
  )

ggsave("benchmark.pdf")

#-----------------------------------------------------------------------------------------------------
#WE TRAIN THE MODELS ON THE TASK
#-----------------------------------------------------------------------------------------------------
#random forest
lrn_ranger <- GraphLearner$new(preprocess_graph %>>% lrn("classif.ranger", predict_type = "prob"))

#XGboost
lrn_xgb    <- GraphLearner$new(preprocess_graph %>>% lrn("classif.xgboost", predict_type = "prob"))

#Logistic regression
lrn_log    <- GraphLearner$new(preprocess_graph %>>% lrn("classif.glmnet", predict_type = "prob"))

#SVM
lrn_svm    <- GraphLearner$new(preprocess_graph %>>% lrn("classif.svm", predict_type = "prob"))

# train the model
#install additional relevant libraries

library(mlr3viz)
install.packages('precrec')
library(precrec)
library(mlr3measures)

#train
lrn_ranger$train(task)
lrn_xgb$train(task)
lrn_svm$train(task)
lrn_log$train(task)

# Make predictions
pred_bench_ranger <- lrn_ranger$predict_newdata(test_data)
pred_bench_xgb    <- lrn_xgb$predict_newdata(test_data)
pred_bench_svm    <- lrn_svm$predict_newdata(test_data)
pred_bench_log    <- lrn_log$predict_newdata(test_data)

#compute logloss
logloss_bench <- c(
  ranger  = mlr3measures::logloss(pred_bench_ranger$truth, pred_bench_ranger$prob),
  xgboost = mlr3measures::logloss(pred_bench_xgb$truth, pred_bench_xgb$prob),
  svm     = mlr3measures::logloss(pred_bench_svm$truth, pred_bench_svm$prob),
  log_reg = mlr3measures::logloss(pred_bench_log$truth, pred_bench_log$prob)
)
logloss_bench

#Make predictions (probabilities) on test data
#pred_bench_ranger <- benchmark_train$predict_newdata(test_data)
#pred_bench_ranger$confusion

#logloss_bench <- pred_bench_ranger$score(msr("classif.logloss"))
#print(logloss_bench)

# Extract predicted probabilities
prob_bench_ranger <- pred_bench_ranger$prob
prob_bench_ranger

#============================================================================================
# STEP 9B:PCA PREPROCESSING PIPELINE
# ===========================================================================================

# PCA is only designed for continuous numerical variables

graph_pca_robust <- 
  po("removeconstants") %>>%
  # Handle categorical features first
  po("imputeoor", affect_columns = selector_type(c("factor", "character", "logical"))) %>>%
  po("encode", method = "one-hot") %>>%
  # Handle numeric features  
  po("imputemedian", affect_columns = selector_type("numeric")) %>>%
  po("pca", rank. = 20)

# Add class balancing before PCA
preprocess_pca_graph <- po("classbalancing",
                           id = "balancer",
                           adjust = "major",
                           reference = "minor",
                           ratio = 4,
                           shuffle = TRUE) %>>%
  graph_pca_robust
# ===========================================================================================
# LEARNERS WITH PCA PREPROCESSING
# ===========================================================================================
learners_pca <- list(
  # Baseline
  GraphLearner$new(
    preprocess_pca_graph %>>% lrn("classif.featureless", predict_type = "prob"),
    id = "baseline_pca"
  ),
  
  # Ranger
  GraphLearner$new(
    preprocess_pca_graph %>>% 
      lrn("classif.ranger", predict_type = "prob",
          num.trees = 100, mtry.ratio = 0.5),
    id = "ranger_pca"
  ),
  
  # XGBoost
  GraphLearner$new(
    preprocess_pca_graph %>>%
      lrn("classif.xgboost", predict_type = "prob",
          nrounds = 100, eta = 0.1, scale_pos_weight = 20),
    id = "xgboost_pca"
  ),
  
  # Penalized Logistic Regression (Elastic Net)
  GraphLearner$new(
    preprocess_pca_graph %>>%
      po("scale") %>>%
      lrn("classif.cv_glmnet", predict_type = "prob",
          alpha = 0.5, nfolds = 5),
    id = "logistic_pca"
  ),
  
  # SVM
  GraphLearner$new(
    preprocess_pca_graph %>>%
      po("scale") %>>%
      lrn("classif.svm", predict_type = "prob",
          type = "C-classification"),
    id = "svm_pca"
  )
)

# ==========================================================================================
# STEP 9A: BENCHMARK EVALUATION
# ==========================================================================================

# Create benchmark design
design_all <- benchmark_grid(
  tasks = task,
  learners = c(learners_bench,learners_pca),
  resamplings = resampling
)

# Run the benchmark
bmr_all <- benchmark(design_all)

# Aggregate and inspect results
results <- bmr_all$aggregate(measures=msr("classif.logloss"))
print(results)

bmr_all %>% autoplot()
bmr_all %>% 
  autoplot() +
  labs(
    x = "Learner",
    y = "Log Loss")
ggsave("pca.pdf")

#our models without PCA consistently perform better (lower log loss)

# ==========================================================================================
# STEP 10: FEATURE SELECTION
# ==========================================================================================

#Start feature selection
install.packages("FSelectorRcpp")
library(FSelectorRcpp)
library(mlr3filters)
library(mlr3tuning)    #Autotuner
library(paradox)       # For defining search space

#Remove one missing observation
train_fs <- train_data[!is.na(stress_ratio)]

# convert target to factor (required for TaskClassif)
train_fs$failurereason <- as.factor(train_fs$failurereason)

# Verify the conversion worked
str(train_fs$failurereason)
levels(train_fs$failurereason)

#--------------------------------------------------------------------------------------------------------
#Clean the target variable but keep it as factor for mlr3
#cat("Original failurereason levels:\n")
#print(levels(data_clean$failurereason))

# Clean level names (remove spaces, special chars)
#levels(train_fs$failurereason) <- make.names(levels(train_fs$failurereason))
#data_clean$failurereason <- droplevels(data_clean$failurereason)

#cat("Cleaned failurereason levels:\n")
#print(levels(data_clean$failurereason))
#print(table(data_clean$failurereason))
#----------------------------------------------------------------------------------------------------

#Name the task
task_fs <- TaskClassif$new(
  id = "wheel_failure_fs",
  backend = train_fs,
  target = "failurereason"
)

#task_fs$positive <- "1"
task_fs %>% autoplot()
task_fs$missings()
task_fs$feature_names

train_fs %>% dim()
#...........................................................................................
# Method 1: Filter-based feature selection using importance
#...........................................................................................

# filter method
base_learner <- lrn("classif.ranger", 
                    num.trees = 50,
                    predict_type = "prob",
                    importance = "impurity")

filter_imp <- flt("importance", learner = base_learner )
filter_imp$calculate(task_fs)
filter_imp$scores

# Create graph with pipeline with filter + learner for importance filter
graph_filter_learner <- as_learner(po("filter", flt("importance", learner = base_learner), 
                                      param_vals = list(filter.nfeat = 10)) %>>% 
                                     base_learner)

#top features
top_imp <- head(as.data.table(filter_imp), 20)
top_imp


#...................................................................................................
#Method 2: wrapper-based feature selection using random forest
#...................................................................................................
set.seed(442)

inner_resampling <- rsmp("cv", folds = 3)
measure_fs <- msr("classif.logloss")
terminator_fs <- trm("evals", n_evals = 50)

afs_random <- AutoFSelector$new(
  learner = base_learner,
  resampling = inner_resampling,
  measure = measure_fs,
  terminator = terminator_fs,
  fselector = fs("random_search")
)
afs_random$id <- "random_ranger"

afs_genetic <- AutoFSelector$new(
  learner = base_learner,
  resampling = inner_resampling,
  measure = measure_fs,
  terminator = terminator_fs,
  fselector = fs("genetic_search",
                 popSize = 10L,
                 elitism = 2L,
                 zeroToOneRatio = 2L)
)
afs_genetic$id <- "genetic_ranger"

outer_resampling <- rsmp("cv", folds = 3)

learners_fs <- list(afs_random, afs_genetic, graph_filter_learner)

design_fs <- benchmark_grid(
  tasks = task_fs,
  learners = learners_fs,
  resamplings = outer_resampling
)

bmr_fs <- benchmark(design_fs, store_models = TRUE)

#-------------------------------------------------------------------------------------------------------------
#comparing filter and wrapper selected features
#-------------------------------------------------------------------------------------------------------------

bmr_fs$score(measure_fs)

autoplot(bmr_fs, type = "boxplot") + ggtitle("Wrapper Feature Selection")
         
ggsave("wrapperfs_vs_importance.pdf")
#random forest using genetic search and random forest using random search performs equivalently the same and well


# filter by 'Importance' 
top_imp <- as.data.table(filter_imp)[order(-score)][1:15, .(feature, importance = score)]
top_imp

# extracting features from each wrapper method

# Train each AutoFSelector individually
afs_random$train(task_fs)
afs_genetic$train(task_fs)

#view features from each wrapper
features_random_search <- afs_random$fselect_result$features
features_random_search
features_genetic_search <- afs_genetic$fselect_result$features
features_genetic_search

# Final Features Selection
# Remove unwanted features and select top 14
features_to_exclude <- c("equipmentnumber", "axlesequencenumber", "failedin30days")
top_features <- features_random_search[[1]][!features_random_search[[1]] %in% features_to_exclude]
top_features <-unlist(top_features)


# Align selected features with actual task feature names
valid_features <- intersect(top_features, task_fs$feature_names)

#------------------------------------------------------------------------------------------------
#BENCHMARKING MODEL ON REDUCED FEATURES
#------------------------------------------------------------------------------------------------

# Create new back end with top_features and create new task
newdata_fs <- task_fs$data(cols = c(valid_features, task_fs$target_names))
sum(is.na(newdata_fs))
dim(newdata_fs) #22273    16
task_top10 <- TaskClassif$new(id = "task_top10", backend = newdata_fs, target = task_fs$target_names)


inner_rsmp <- rsmp("cv", folds = 5)
measure <- msrs("classif.logloss")
terminate <- trm( "evals", n_evals = 100)

# Create benchmark design on reduced features 
design_new_features <- benchmark_grid(
  tasks = task_top10,
  learners = learners_bench,
  resamplings = rsmp("cv", folds = 5)
)

# Run the benchmark
bmr_new_features <- benchmark(design_new_features)

# Aggregate and inspect results
results <- bmr_new_features$aggregate(msrs("classif.logloss"))
print(results)
#The Logistic regression and xgboost have the lowest log loss

bmr_new_features %>% 
  autoplot() +
  labs(
    x = "Learner",
    y = "Log Loss"
  )

ggsave("benchmark_new_features.pdf")

#-----------------------------------------------------------------------------------------------------
#TRAINING THE MODELS ON THE NEW TASK
#-----------------------------------------------------------------------------------------------------

lrn_ranger$train(task_top10)
lrn_xgb$train(task_top10)
lrn_svm$train(task_top10)
lrn_log$train(task_top10)

# Make predictions
pred_fs_ranger <- lrn_ranger$predict_newdata(test_data)
pred_fs_xgb    <- lrn_xgb$predict_newdata(test_data)
pred_fs_svm    <- lrn_svm$predict_newdata(test_data)
pred_fs_log    <- lrn_log$predict_newdata(test_data)

#compute logloss
logloss_new_features <- c(
  ranger  = mlr3measures::logloss(pred_fs_ranger$truth, pred_fs_ranger$prob),
  xgboost = mlr3measures::logloss(pred_fs_xgb$truth, pred_fs_xgb$prob),
  svm     = mlr3measures::logloss(pred_fs_svm$truth, pred_fs_svm$prob),
  log_reg = mlr3measures::logloss(pred_fs_log$truth, pred_fs_log$prob)
)
logloss_new_features


# Extract predicted probabilities
prob_fs_ranger <- pred_fs_ranger$prob
prob_fs_ranger


#===========================================================================================
# STEP 11: HYPERPARAMETER TUNING FOR XGBOOST
#===========================================================================================

#library(mlr3misc)
library(mlr3mbo)
library(bbotk)
#library(DiceKriging)
#library(rgenoud)

# Hyperparameter search space
xgb_params <- ps(
  classif.xgboost.nrounds = p_int(50, 300),
  classif.xgboost.eta = p_dbl(0.01, 0.3),
  classif.xgboost.max_depth = p_int(3, 10),
  classif.xgboost.subsample = p_dbl(0.7, 1.0),
  classif.xgboost.colsample_bytree = p_dbl(0.7, 1.0),
  classif.xgboost.scale_pos_weight = p_int(10, 30)
)

# Random search tuner
terminator_tune <- trm("evals", n_evals = 30)
xgb_tuner <- tnr("random_search")

# Define AutoTuner for XGBoost
at_xgb <- AutoTuner$new(
  tuner = xgb_tuner,
  learner = lrn_xgb,
  resampling = rsmp("cv", folds = 3),
  measure = msr("classif.logloss"),
  search_space = xgb_params,
  terminator = terminator_tune
)

# Benchmark tuned XGBoost
design_hpo <- benchmark_grid(
  tasks = task_top10,
  learners = at_xgb,
  resamplings = outer_resampling
)

# Run benchmark
bmr_hpo <- benchmark(design_hpo)

# Aggregate results
hpo_results <- bmr_hpo$aggregate(msrs(c("classif.ce", "classif.logloss", "classif.bacc")))
print(hpo_results)

# Plot performance distribution
autoplot(bmr_hpo, type = "boxplot", measure = msr("classif.logloss"))
ggsave("xgb_hpo_boxplot.pdf")

#===========================================================================================
# STEP 12: FINAL MODEL EVALUATION ON TEST SET
#===========================================================================================

# Train tuned XGBoost model on full training data (with reduced features if applied earlier)
at_xgb$train(task_top10)

# Predict on held-out test set
pred_xgb <- at_xgb$predict_newdata(test_data)

# Print confusion matrix
print(pred_xgb$confusion)

# Compute performance metrics on test data
logloss_test <- pred_xgb$score(msr("classif.logloss"))


# Extract predicted probabilities
prob_xgb <- pred_xgb$prob
head(prob_xgb)
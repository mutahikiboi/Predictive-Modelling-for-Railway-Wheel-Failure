# Predictive-Modelling-for-Railway-Wheel-Failure
2025 Railway Application Section (RAS) Problem Solving Competition to predict railway wheel failures within the next 30 days using heterogeneous sensor data, railcar attributes, and historical failure records.

This repository contains the code and materials for a predictive maintenance project developed for the 2025 Railway Application Section (RAS) Problem Solving Competition. The goal of the project is to predict railway wheel failures within the next 30 days using heterogeneous sensor data, railcar attributes, and historical failure records.

The work combines extensive data preprocessing, feature engineering, model benchmarking, hyperparameter optimization, and interpretability analysis, with a strong focus on handling class imbalance and probabilistic evaluation (log-loss).

## Project Overview

Wheelsets are critical safety components in railway operations. Failures such as high flange, thin flange, or high impact can cause infrastructure damage and increase derailment risk. Using data from multiple detector systems, this project builds machine learning models to estimate the probability of wheel failure in a short-term (30-day) window.

## Key objectives:

Predict whether a wheel will fail within 30 days

Identify likely failure types (multi-class setting)

Compare multiple ML models under a unified benchmarking framework

Provide interpretable insights into the drivers of wheel failure

The dataset spans approximately 7 years and integrates multiple railway monitoring systems:

## Source	Description
WPD	Wheel Profile Detector (geometry and wear metrics)
THD	Truck Hunting Detectors (lateral dynamics)
WILD	Wheel Impact Load Detectors (vertical force measurements)
Mileage	Aggregated loaded and empty travel distances
Failures	Failure labels and reasons (30-day horizon)

# Feature Engineering

In addition to ~43 raw features, 13 domain-informed engineered features were created to better capture operational stress and degradation patterns, including:

Equipment age (days)

Cumulative and total travel mileage

Utilization ratio (loaded vs total travel)

Average operational speed

Dynamic loading score (composite stress indicator)

Shock stress and stress ratio

Temporal features (applied year/month, applied age)

These features were designed to reflect cumulative usage, dynamic stress, and temporal degradation.

# Feature Selection

To reduce dimensionality and improve generalization, multiple feature selection strategies were applied:

Filter-based: Random Forest impurity importance

Wrapper-based:

Random search

Genetic algorithm search

Wrapper-based approaches consistently outperformed filters, resulting in a final subset of 14 features that balanced predictive power and interpretability.

Modeling Approach

All models were implemented using the mlr3 ecosystem in R, with unified preprocessing pipelines:

Models Benchmarked

Featureless baseline

Random Forest (ranger)

XGBoost

Logistic Regression (Elastic Net)

Support Vector Machine (RBF kernel)

Key Design Choices

5-fold cross-validation

Log-loss as the primary evaluation metric

Class imbalance handling via targeted class balancing

Optional PCA evaluated but ultimately discarded due to performance degradation

## Best-performing models:

Random Forest (most stable generalization)

XGBoost (competitive after hyperparameter optimization)

Hyperparameter Optimization

XGBoost was further optimized using:

Random search

Bayesian optimization (mlr3mbo)

Optimized parameters included:

Number of boosting rounds

Learning rate

Tree depth

Sampling ratios

Class imbalance weighting

Interpretability

To ensure transparency and trustworthiness, multiple interpretability techniques were applied:

Global Feature Importance (cross-entropy based)

Local explanations (LIME-style)

SHAP values (game-theoretic feature attribution)

Key drivers of failure risk included:

Total travel miles

Average speed

Flange thickness

Dynamic vertical loading

Applied age (temporal exposure)

No evidence of label leakage was detected during interpretability analysis.

Wheel failures are rare and highly imbalanced (<2%)

Tree-based ensemble methods outperform linear models

Feature engineering and wrapper-based selection significantly improve log-loss

Binary failure prediction is more reliable than fine-grained multi-class prediction

This project is released under the MIT License
# DyadAnalysis
**Analysis of dyadic data from behavioral interactions**

Junjie Han, Janice Siegford, Gustavo de los Campos, Robert Tempelman, Cedric Gondro, and Juan Steibel

If you find this resource helpful, please cite.

This repository shows the implementation of dyadic data analysis by fitting generalized linear mixed models. A Bayesian approach is utilized to estimate the model parameters.

## Table of contents

* [Software](#Software)
* [Code](#Code)
* [Dataset](#Dataset)

## Software
* All experiments are implemented in [R](https://cloud.r-project.org/) (version 4.1.2)

* The model is fitted using a Bayesian approach that is implemented with [rstan](https://mc-stan.org/users/interfaces/rstan) package in R

* [Bayesplot](https://mc-stan.org/users/interfaces/bayesplot) is used for posterior checking of the fitted Bayesian model

## Code
* R code and the stan script for model compilation can be found in the __Code__ folder

* Notice that there are two R script files and two stan script files:

`runBayes.R` is the high-level R script whose functionality includes: 1) preparing data, 2) calling stan model, 3) calling stan output variable, and 4) posterior predictive checking.

`hurdle.stan` is the model-level stan script that allows the user to compile/modify the model fitting using the Bayesian approach. Customized quantaties/metrics are editable in this script.


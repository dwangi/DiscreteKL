library(Rcpp)
library(tidyr)
library(discSurv)
library(mvtnorm)
library(devtools)
devtools::install_github("dwangi/DiscreteKL")

#source('~/DESKTOP/DiscreteKL_pkg/Discrete_data.R')
#sourceCpp("~/DESKTOP/DiscreteKL_pkg/Boosting/DiscreteKL_Boosting_logit.cpp")
#sourceCpp("~/DESKTOP/DiscreteKL_pkg/NR/DiscreteKL_NR_logit.cpp")
#source("~/DESKTOP/DiscreteKL_pkg/NR/DiscreteKL_NR_logit.R")
#source('~/DESKTOP/DiscreteKL_pkg/Boosting/DiscreteKL_Boosting_logit.R')
library(DiscreteKL)

start <- Sys.time()
rep <- 1
loglik <- as.data.frame(matrix(rep(0, (3*rep)), rep, 3))
names(loglik) <- c("NR", "Boosting", "KL_boosting")
eta <- as.data.frame(matrix(rep(0, (1*rep)), rep, 1))
names(eta) <- c("KL_boosting")

n_internal = 200
n_prior = 10000
p = 10
p_i = 2
corr = 0.75
p_true = 5

M <- 1000
Z.char <- paste0('Z', 1:p)
internal_beta <- rep(0, p)
test_beta <- rep(0, p)
prior_beta <- rep(0, p)

#internal/test
internal_index <- c(1,3,5,7,9)
test_index <- internal_index
#prior
prior_index <- c(1,3,5,6,8)

internal_beta[internal_index] =c(0.08, 0.08, -0.08, -0.08, 0.08) 
test_beta[test_index] =c(0.08, 0.08, -0.08, -0.08, 0.08)     
prior_beta[prior_index]=c(0.08, 0.08, -0.08, -0.08, 0.08)   

day_effect <- c(-0.01, -0.02, -0.03, -0.04, -0.05)

#Simulate prior data using prior 0
set.seed(220)
prior_data <- sim.disc.block.ar1(prior_beta, day_effect, n_prior, p_i, corr, Z.char)

###### estimate prior beta######
X_prior <- as.matrix(prior_data[, prior_index])
betap=discSurv_logit(prior_data$time, X_prior, prior_data$status)
prior_betav <- as.vector(betap$beta_v)
beta_t_prior  <- as.vector(betap$beta_t)

prior_beta[prior_index] <- prior_betav
prior_beta
################################################################3
internal_beta <- as.matrix(internal_beta)
prior_beta <- as.matrix(prior_beta)

start_time <- Sys.time()

for (i in 1:rep){
  set.seed(i)
  ############# block.ar1 ####################
  internal_data <- sim.disc.block.ar1(internal_beta, day_effect, n_internal, p_i, corr, Z.char)
  test_data <- sim.disc.block.ar1(internal_beta, day_effect, n_internal, p_i, corr, Z.char)
  mean(internal_data$status)  #0.8
  X_internal <- as.matrix(dplyr::select(internal_data, -c("time", "status")))
  delta_internal <- internal_data$status
  t_internal <- internal_data$time
  
  X_test <- as.matrix(dplyr::select(test_data, -c("time", "status")))
  delta_test <- test_data$status
  t_test <- test_data$time
  #############NR
  result_NR=discSurv_logit(internal_data$time, X_internal, internal_data$status)
  result_NR$beta_t
  result_NR$beta_v
  loglik$NR[i] <- DiscLoglik_logit(test_data$time, X_test, test_data$status,result_NR$beta_t,result_NR$beta_v)
    
  #############Boosting
  result_boosting <- discrete_boosting_logit(t_internal, X_internal, delta_internal, tol=1e-10, Mstop=M, step_size=0.005)
  result_boosting$beta_t
  result_boosting$beta_v
  result_boosting$m
  
  loglik$Boosting[i] <- DiscLoglik_logit(t_test, X_test, delta_test, result_boosting$beta_t, result_boosting$beta_v)
    
  #############KL_boosting1
  KL_boosting <- KL_logit_Boosting(beta_t_prior, prior_beta, 0, 10, 0.25, internal_data,M, step_size=0.005)
  result_KL <- KL_boosting$model
  result_KL$beta_t
  result_KL$beta_v
  result_KL$m
    
  eta$KL_boosting[i] <- KL_boosting$eta
  loglik$KL_boosting[i] <- DiscLoglik_logit(t_test, X_test, delta_test, result_KL$beta_t, result_KL$beta_v)
}
deviance <- -loglik
deviance
eta$KL_boosting





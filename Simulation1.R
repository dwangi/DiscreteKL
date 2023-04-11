library(mvtnorm)
library(discSurv)
library(survival)
library(tidyr)
library(Rcpp)
library(parallel)
library(devtools)
devtools::install_github("dwangi/DiscreteKL")
library(DiscreteKL)
###### this simulation code can be applied with parallel computing
##### detect max number of cores on this machine
max.cores = parallel::detectCores()
##### by default, we will use floor(0.5*max.cores) to conduct the analysis 
####### internal sample size and censoring rate
####### censoring rate is set as 40%
####### sample size is 150 or 200
n_local = 150
cens_rate = 40
#### number of simulation replicates
#here we set it as 50, note that in the paper we replicated the simulations 500 times
rep = 50
length.out = 500

Deviance_internal <- rep(0, rep)
eta <- as.data.frame(matrix(rep(0, 1*rep), rep, 1))
beta_KL <- as.data.frame(matrix(rep(0, 20*rep), rep, 20))
beta_internal <- as.data.frame(matrix(rep(0, 20*rep), rep, 20))
Deviance_KL <- rep(0, (length.out+1))
Bias_KL <- rep(0, (length.out+1))
SE_KL <- rep(0, (length.out+1))
MSE_KL <- rep(0, (length.out+1))

n_local = n_local
n_prior = 10000
n_test = 1000
p_local = 10
p_prior = 10

if (cens_rate==40){
  cens_upper = 11
}
if (cens_rate==20){
  cens_upper = 1000
}

Z.char_prior <- paste0('Z', 1:p_prior)
Z.char_local <- paste0('Z', 1:p_local)

#prior and local data parameters
local_beta <- c(2,-1,2,3,-1,4,-1,3,4,-1)
external_beta <- local_beta
day_effect <- c(-6.0, -6.0, -6.0, -4.5, -4.5,
                -4.5, -3.0, -3.0, -1.5, -1.5)

#simulate prior data assuming local_beta is the beta for true model
prior_data <- sim.disc(local_beta, day_effect, n_prior, Z.char_prior, 1000)
1-mean(prior_data$status)
X_prior <- dplyr::select(prior_data, -c("time", "status", "Z9", "Z10"))
X_prior <- as.matrix(X_prior)

betap=discSurv_logit(prior_data$time, X_prior, prior_data$status)
prior_beta <- as.vector(c(betap$beta_v[1:8],0,0))
prior_day_effect <- as.vector(betap$beta_t)
betap$iter

local_beta <- as.matrix(local_beta)
external_beta <- as.matrix(external_beta)
prior_beta <- as.matrix(prior_beta)

data_internal_list <- vector(mode="list", length=rep)

#true model parameters
beta <- c(local_beta,day_effect)

KL_lambda <- function(x){
  Deviance_KL <- rep(0, rep)
  beta_KL <- as.data.frame(matrix(rep(0, 20*rep), rep, 20))
  for (i in 1:rep){
    local_data <- data_internal_list[[i]]$data
    Z_local <- dplyr::select(local_data, -c("time", "status"))
    data_test <- data_internal_list[[i]]$data_test
    Z_test <- dplyr::select(data_test, -c("time", "status"))
    estKL <- discSurvKL_logit(local_data$time, Z_local, local_data$status, tol = 1e-20, max_iter = 25, prior_day_effect, prior_beta, eta=x)
    beta_KL[i,] <- as.numeric(c(estKL$beta_v,estKL$beta_t))
    Deviance_KL[i] <- -DiscLoglik_logit(data_test$time, Z_test, data_test$status,estKL$beta_t,estKL$beta_v)
  }
  
  mean_Deviance_KL <- mean(replace(Deviance_KL, Deviance_KL == 0, NA), na.rm = TRUE)
  mean_beta_KL <- colMeans(replace(beta_KL, beta_KL == 0, NA), na.rm = TRUE)
  mean_Bias_KL <- mean(abs(mean_beta_KL-beta))
  ESE_KL <- mean(apply(replace(beta_KL, beta_KL == 0, NA), MARGIN = 2, function(x){sd(x, na.rm = TRUE)}))
  MSE_KL <- mean_Bias_KL^2+ESE_KL^2
  
  return_list <- list("Deviance" = mean_Deviance_KL, "Bias" = mean_Bias_KL, 
                      "SE" = ESE_KL, "MSE" = MSE_KL)
  return(return_list)
}

for (i in 1:rep){
  set.seed(i)
  data_internal_list[[i]]$data <- sim.disc(local_beta, day_effect, n_local, Z.char_local, cens_upper)
  data_internal_list[[i]]$data_test <- sim.disc(external_beta, day_effect, n_test, Z.char_local, 1000)
}

eta_list <- vector(mode="list", length=(length.out+1))

for (i in 0:length.out){
  eta_list[[i+1]] <- 0.01*i
}

models_KL <- mclapply(eta_list, KL_lambda, mc.cores = floor(0.5*max.cores))

#################################
for (i in 1:rep){
set.seed(i)
#############Prior
local_data <- data_internal_list[[i]]$data
1-mean(local_data$status)
external_data <- data_internal_list[[i]]$data_test

Z_local <- dplyr::select(local_data, -c("time", "status"))
Z_external <- dplyr::select(external_data, -c("time", "status"))

#local
estLocal <- discSurv_logit(local_data$time, Z_local, local_data$status)
Deviance_internal[i] <- -DiscLoglik_logit(external_data$time, Z_external, external_data$status,estLocal$beta_t,estLocal$beta_v)
beta_internal[i,] <- as.numeric(c(estLocal$beta_v,estLocal$beta_t))
}

mean_Deviance_internal <- mean(replace(Deviance_internal, Deviance_internal == 0, NA), na.rm = TRUE)
mean_Deviance_internal <- mean_Deviance_internal/n_test
mean_beta_internal <- colMeans(replace(beta_internal, beta_internal == 0, NA), na.rm = TRUE)
mean_Bias_internal <- mean(abs(mean_beta_internal-beta))
ESE_internal <- mean(apply(replace(beta_internal, beta_internal == 0, NA), MARGIN = 2, function(x){sd(x, na.rm = TRUE)}))
MSE_internal <- mean_Bias_internal^2+ESE_internal^2

for (i in 1:(length.out+1)){
  {
    results_KL <- models_KL[[i]]
    Bias_KL[i] <- results_KL$Bias
    SE_KL[i] <- results_KL$SE
    MSE_KL[i] <- results_KL$MSE
    Deviance_KL[i] <- results_KL$Deviance/n_test
  }
}

### Figs
x_lim_up <- 5
if (n_local==200 && cens_rate==20){
  x_lim_up <- 1
}

#1
#Deviance
library(colorspace)
cols <- c("#fcd575", "#6C71C2")
library(ggplot2)

mr <- data.frame(Method = c(
  rep("KL", (length.out+1)),
  rep("Internal", (length.out+1))),
  loglik = c(
    Deviance_KL,
    rep(mean_Deviance_internal, (length.out+1))),
  eta = c(seq(0,5,length.out=(length.out+1)), seq(0,5,length.out=(length.out+1)))
)
mr$Method <- factor(mr$Method,
                    levels = c('Internal', 'KL'),ordered = TRUE)
g<-ggplot(mr, aes(x=eta, y=loglik, group=Method)) + 
  geom_line(aes(linetype=Method, color=Method, size = Method)) +
  xlim(0, x_lim_up) +
  geom_point(x=0, y=mean_Deviance_internal, color="#fcd575", size=5)+
  labs(y = "Predictive Deviance", x = bquote(omega))+  scale_color_manual(values=cols)+ theme_bw() + 
  scale_linetype_manual(values=c("dotted", "solid"))+scale_size_manual( values = c(1,2) ) +
  theme(panel.border = element_blank(), 
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))+theme(axis.title = element_text(size = 14))+theme(axis.text = element_text(size = 14))+theme(legend.position = "none")
print(g)

#MSE
library(colorspace)
cols <- c("#fcd575", "#6C71C2")
library(ggplot2)

mr <- data.frame(Method = c(
  rep("KL", (length.out+1)),
  rep("Internal", (length.out+1))),
  loglik = c(
    MSE_KL,
    rep(MSE_internal, (length.out+1))),
  eta = c(seq(0,5,length.out=(length.out+1)), seq(0,5,length.out=(length.out+1)))
)
mr$Method <- factor(mr$Method,
                    levels = c('Internal', 'KL'),ordered = TRUE)
g<-ggplot(mr, aes(x=eta, y=loglik, group=Method)) + 
  geom_line(aes(linetype=Method, color=Method, size = Method)) + 
  xlim(0, x_lim_up) +
  geom_point(x=0, y=MSE_internal, color="#fcd575", size=5)+
  labs(y = "MSE", x = bquote(omega))+  scale_color_manual(values=cols)+ theme_bw() + scale_linetype_manual(values=c("dotted", "solid"))+scale_size_manual( values = c(1,2) )+
  theme(panel.border = element_blank(), 
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))+theme(axis.title = element_text(size = 14))+theme(axis.text = element_text(size = 14))+theme(legend.position = "none")
print(g)









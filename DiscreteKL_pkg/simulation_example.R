library(mvtnorm)
library(discSurv) #used to simulate data
library(tidyr)
library(Rcpp)

rep=10
source('~/DESKTOP/DiscreteKL_pkg/NR/DiscreteKL_NR_logit.R')
source('~/DESKTOP/DiscreteKL_pkg/NR/Discrete_data.R')
sourceCpp("~/DESKTOP/DiscreteKL_pkg/NR/DiscreteKL_NR_logit.cpp")

loglik <- as.data.frame(matrix(rep(0, 4*rep), rep, 4))
names(loglik) <- c("KL_logit", "prior", "local", "stacked")
eta <- as.data.frame(matrix(rep(0, rep), rep, 1))
names(eta) <- c("KL_prior")

n_local = 300
n_prior = 10000
p_local = 10
p_prior = 10
cens_upper = 11 #90% censoring

Z.char_prior <- paste0('Z', 1:p_prior)
Z.char_local <- paste0('Z', 1:p_local)

#prior and local data parameters
local_beta <- c(-2,1,-2,-3,1,-4,1,-3,-4,1)
external_beta <- local_beta

day_effect <- c(-6,-6,-6,-4.5,-4.5,-4.5,-3,-3,-1.5,-1.5)

#simulate prior data assuming local_beta is the beta for true model
prior_data <- sim.disc(local_beta, day_effect, n_prior, Z.char_prior, cens_upper)
1-mean(prior_data$status) #~90%

#Setting b
X_prior <- dplyr::select(prior_data, -c("time", "status", "Z9", "Z10"))
X_prior <- as.matrix(X_prior)
betap=discSurv_logit(prior_data$time, X_prior, prior_data$status)
prior_beta <- as.vector(c(betap$beta_v[1:8],0,0))
prior_day_effect <- as.vector(betap$beta_t)
local_beta <- as.matrix(local_beta)
external_beta <- as.matrix(external_beta)
prior_beta <- as.matrix(prior_beta)
#################################
for (i in 1:rep){
set.seed(i)
#############Prior
local_data <- sim.disc(local_beta, day_effect, n_local, Z.char_local, cens_upper)
1-mean(local_data$status)
external_data <- sim.disc(external_beta, day_effect, n_local, Z.char_local, cens_upper)

Z_local <- dplyr::select(local_data, -c("time", "status"))
Z_external <- dplyr::select(external_data, -c("time", "status"))

#prior
loglik$prior[i] <- DiscLoglik_logit(external_data$time, Z_external, external_data$status,prior_day_effect,prior_beta)

#local
estLocal <- discSurv_logit(local_data$time, Z_local, local_data$status)
loglik$local[i] <- DiscLoglik_logit(external_data$time, Z_external, external_data$status,estLocal$beta_t,estLocal$beta_v)

#KL
KL_prior <- kl_logit(prior_day_effect, prior_beta, 0, 10, 0.25, local_data)
estKL <- KL_prior$model
eta$KL_prior[i] <- KL_prior$eta
loglik$KL_logit[i] <- DiscLoglik_logit(external_data$time, Z_external, external_data$status,estKL$beta_t,estKL$beta_v)

#stacked regression: only beta from the prior model: estimate day effect
Z_local_stacked <- data.frame(Z_stacked = as.matrix(Z_local[,1:p_prior])%*%as.matrix(prior_beta), local_data$Z9, local_data$Z10)
Z_external_stacked <- data.frame(Z_stacked = as.matrix(Z_external[,1:p_prior])%*%as.matrix(prior_beta), external_data$Z9, external_data$Z10)
estStacked <- discSurv_logit(local_data$time, Z_local_stacked, local_data$status)
loglik$stacked[i] <- DiscLoglik_logit(external_data$time, Z_external_stacked, external_data$status,estStacked$beta_t,estStacked$beta_v)
}

deviance <- -loglik
colMeans(deviance)

library(colorspace)
cols <- c("darkgoldenrod1", "chartreuse3", "cornflowerblue", "mediumorchid2")
library(ggplot2)
mr <- data.frame(Method = c(
  rep("KL", rep),
  rep("External", rep),
  rep("Internal", rep),
  rep("Stacked", rep)),
  loglik = c(
    deviance$KL_logit,
    deviance$prior,
    deviance$local,
    deviance$stacked))
mr$Method <- factor(mr$Method,
                    levels = c('Internal','KL', 'Stacked','External'),ordered = TRUE)
g<-ggplot(mr, aes(x=Method, y=loglik, fill=Method)) +
  geom_boxplot() + labs(y = "Predictive Deviance", x = "Method")+  scale_colour_manual(
    values = cols,
    aesthetics = c("fill")
  )+ theme_bw() + theme(panel.border = element_blank(), panel.grid.major = element_blank(),
                        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))+
  theme(axis.title = element_text(size = 15))+theme(axis.text = element_text(size = 15))+
  theme(legend.position = "none")+theme(axis.title.x = element_blank())
print(g)

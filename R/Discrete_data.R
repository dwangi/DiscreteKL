AR1 <- function(tau, m) {
  if(m==1) {R <- 1}
  if(m > 1) {
    R <- diag(1, m)
    for(i in 1:(m-1)) {
      for(j in (i+1):m) {
        R[i,j] <- R[j,i] <- tau^(abs(i-j))
      }
    }
  }
  return(R)
}

simu_z <- function(n, size.groups)
{
  Sigma_z1=diag(size.groups) # =p
  Corr1<-AR1(0.5,size.groups) #correlation structure 0.5 0.6
  diag(Corr1) <- 1
  Sigma_z1<- Corr1
  pre_z= rmvnorm(n, mean=rep(0,size.groups), sigma=Sigma_z1)
  return(pre_z)
}

inv.logit <- function(x)
{
  return((exp(x)/(1+exp(x))))
}

sim.disc <- function(beta, eta, prov.size, Z.char, censor) {
  N <- prov.size 
  p_1 <- 0.5*length(beta)
  Z1 <- as.matrix(simu_z(N, p_1))
  Z2 <- matrix(rbinom(N*p_1,1,0.5),N,p_1)
  Z <- cbind(Z1, Z2)
  day <- 1 #1
  idx.atrisk <- 1:N
  days.to.event <- rep(length(eta), N)
  status <- rep(0, N)
  probs <- plogis(eta[1]+(as.matrix(Z)%*%beta))
  idx.event <- idx.atrisk[rbinom(length(probs), 1, probs)==1]
  status[idx.event] <- 1
  days.to.event[idx.event] <- day
  idx.out <- idx.event
  censoring=runif(N,1,censor)
  conTime = data.frame(time=censoring)
  censoring_time <- as.numeric(contToDisc(dataShort = conTime, timeColumn = "time", intervalLimits = 1:(censor))$timeDisc)#3
  for (x in tail(eta,-1)) {
    day <- day+1
    idx.atrisk <- c(1:N)[-idx.out]  
    probs <- plogis(x+(as.matrix(Z[idx.atrisk,])%*%beta))
    idx.event <- idx.atrisk[rbinom(length(probs), 1, probs)==1]
    status[idx.event] <- 1
    days.to.event[idx.event] <- day
    idx.out <- unique(c(idx.out, idx.event))
  }
  
  tcens <- as.numeric(censoring<days.to.event) # censoring indicator
  delta <- 1-tcens
  time <- days.to.event*(delta==1)+censoring_time*(delta==0)
  delta[-idx.out] <- 0
  data <- as.data.frame(cbind(delta, Z, time))
  colnames(data) <- c("status", Z.char, "time")
  
  return(data)
}

#### Block: AR1 ######################################################

simu_z_block_ar1 <- function(n, size.groups, p, corr)
{
  m <- p/size.groups
  Sigma_z1=diag(p)
  for (i in 1:m){
    Corr1<-AR1(corr, size.groups) #correlation structure 0.5 0.6
    diag(Corr1) <- 1
    Sigma_z1[((i-1)*size.groups+1):(i*size.groups), ((i-1)*size.groups+1):(i*size.groups)]<- Corr1
  }
  pre_z= rmvnorm(n, mean=rep(0,p), sigma=Sigma_z1)
  pre_z=scale(pre_z)
  return(pre_z)
}

sim.disc.block.ar1 <- function(beta, eta, prov.size, p_i, corr, Z.char) {
  N <- prov.size 
  p <- length(beta)
  Z <- simu_z_block_ar1(N, p_i, p, corr)
  day <- 1 #1
  idx.atrisk <- 1:N
  days.to.event <- rep(length(eta), N)
  status <- rep(0, N)
  probs <- plogis(eta[1]+(as.matrix(Z)%*%beta))
  idx.event <- idx.atrisk[rbinom(length(probs), 1, probs)==1]
  status[idx.event] <- 1
  days.to.event[idx.event] <- day
  idx.out <- idx.event
  censoring=runif(N,1,3*length(eta))
  conTime = data.frame(time=censoring)
  censoring_time <- as.numeric(contToDisc(dataShort = conTime, timeColumn = "time", intervalLimits = 1:(3*length(eta)))$timeDisc)
  for (x in tail(eta,-1)) {
    day <- day+1
    idx.atrisk <- c(1:N)[-idx.out]  
    probs <- plogis(x+(as.matrix(Z[idx.atrisk,])%*%beta))
    idx.event <- idx.atrisk[rbinom(length(probs), 1, probs)==1]
    status[idx.event] <- 1
    days.to.event[idx.event] <- day
    idx.out <- unique(c(idx.out, idx.event))
  }
  
  tcens <- as.numeric(censoring<days.to.event) # censoring indicator
  delta <- 1-tcens
  time <- days.to.event*(delta==1)+censoring_time*(delta==0)
  delta[-idx.out] <- 0
  data <- as.data.frame(cbind(Z, delta, time))
  colnames(data) <- c(Z.char, "status", "time")
  
  return(data)
}


KL_logit_Boosting <- function(day_prior, beta_prior, eta_min, eta_max, eta_interval, df_input, M, step_size)
{
  eta_vec <- seq(from = eta_min, to = eta_max, by = eta_interval)
  df_input <- as.data.frame(df_input)
  likelihood <- eta_vec - eta_vec
  p_prior <- length(beta_prior)
  Z <- dplyr::select(df_input, -c("time", "status"))
  df_input$LP_prior <- as.matrix(Z[, 1:p_prior])%*%as.matrix(beta_prior)
  
  folds <- cut(seq(1,nrow(df_input)),breaks=5,labels=FALSE)
  
  k=0
  for (eta in eta_vec)
  {
    k=k+1
    likelihood_cv = rep(0, 5)
    for (cv in 1:5)
    {
      testIndexes <- which(folds==cv,arr.ind=TRUE)
      df_input$y <- df_input$status
      df_test <- df_input[testIndexes, ]
      df_train <- df_input[-testIndexes, ]
      X_train <- Z[-testIndexes, ]
      X_test <- Z[testIndexes, ]

      X_train <- as.matrix(X_train)
      delta_train <- df_train$y
      t_train <- df_train$time
      result_train <- discreteKL_boosting_logit(t_train, X_train, delta_train, day_prior, df_train$LP_prior, eta, tol=1e-30, Mstop=M, step_size)
      likelihood_cv[cv] <- DiscLoglik_logit(df_test$time, X_test, df_test$y,result_train$beta_t,result_train$beta_v)
    }
    likelihood[k] <- mean(likelihood_cv)
  }
  max_loc <- which(likelihood==max(likelihood))
  eta_where_max <- eta_vec[max_loc][1]
  
  delta_input <- df_input$y
  t_input <- df_input$time
  Z <- as.matrix(Z)
  result_final <- discreteKL_boosting_logit(t_input, Z, delta_input, day_prior, df_input$LP_prior, eta=eta_where_max, tol=1e-30, Mstop=M, step_size)
  
  return_list <- list("model"=result_final, "eta"= eta_where_max, "likelihood"=likelihood)
  return(return_list)
}

#####cpp version discrete model#################################################
discreteKL_boosting_logit=function(t, X, ind, beta_t_prior, LP_prior, eta, tol, Mstop, step_size){
  maxt=max(t)
  c=ncol(X)
  r=nrow(X)
  beta_t <- rep(0, maxt)
  beta_v <- rep(0, c)
  for (k in 1:maxt){
    delta_sub <- ind[t==k]
    beta_t[k] <- mean(delta_sub)
  }
  
  # order t
  od=order(t,decreasing = T)
  t_od=t[od]
  ind=ind[od]
  LP_prior=LP_prior[od]
  X=X[od,]
  
  # formatting
  ind=as.matrix(ind)
  beta_t1=as.matrix(beta_t)
  beta_v1=as.matrix(beta_v)
  t=as.matrix(t_od)
  beta_t_prior=as.matrix(beta_t_prior)
  LP_prior=as.matrix(LP_prior)
  
  return (NR_boostingKL_logit(t, X, ind,  beta_t_prior, LP_prior, beta_t1, beta_v1, eta, tol, Mstop, step_size))
}

DiscLoglik_logit<-function(t, X, ind, beta_t, beta_v){
  maxt=max(t)
  c=ncol(X)
  r=nrow(X)
  # order t
  od=order(t,decreasing = T)
  t=t[od]
  ind=ind[od]
  X=X[od,]
  
  # formatting
  ind=as.matrix(ind)
  beta_t=as.matrix(beta_t)
  beta_v=as.matrix(beta_v)
  t=as.matrix(t)
  X=as.matrix(X)
  
  return (dbeta_logit(t, X, ind, beta_t, beta_v, maxt, c, r)$loglik)
}

discrete_boosting_logit=function(t, X, ind, tol, Mstop, step_size){
  maxt=max(t)
  c=ncol(X)
  r=nrow(X)
  beta_t <- rep(0, maxt)
  beta_v <- rep(0, c)
  for (k in 1:maxt){
    delta_sub <- ind[t==k]
    beta_t[k] <- mean(delta_sub)
  }
  
  # order t
  od=order(t,decreasing = T)
  t_od=t[od]
  ind=ind[od]
  X=X[od,]
  
  # formatting
  ind=as.matrix(ind)
  beta_t=as.matrix(beta_t)
  beta_v=as.matrix(beta_v)
  t=as.matrix(t_od)
  X=as.matrix(X)
  
  return (boosting_logit(t, X, ind, beta_t, beta_v, tol, Mstop, step_size))
}



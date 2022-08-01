#include <Rcpp.h>
#include <RcppEigen.h>
#include <math.h>
//[[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using Eigen::MatrixXd;
 
  
// [[Rcpp::export]]
List dbetaKL_logit(Eigen::MatrixXd t, Eigen::MatrixXd beta_t_prior, Eigen::MatrixXd LP_prior, Eigen::MatrixXd Z, Eigen::MatrixXd delta, Eigen::MatrixXd beta_t, Eigen::MatrixXd beta_v, int K, int p, int n, double eta){
  double loglik=0;
  Eigen::MatrixXd score_t(K,1);
  score_t.setZero(K,1);
  Eigen::MatrixXd score_v(p,1);
  score_v.setZero(p,1);
  Eigen::MatrixXd info_t(K,1);
  info_t.setZero(K,1);
  
  for (int i = 0 ; i < n ; i++){
    for (int s = 1 ; s <= t(i,0) ; s++){
        double delta_prior=1/(1+exp(-LP_prior(i,0)-beta_t_prior(s-1,0)));
        double lambda=1/(1+exp(-beta_t(s-1,0)-(Z.row(i)*beta_v)(0,0)));
        loglik=loglik+log(1-lambda);
        score_t(s-1,0)=score_t(s-1,0)-lambda;
        score_v=score_v-lambda*Z.row(i).transpose();
        info_t(s-1,0)=info_t(s-1,0)+lambda*(1-lambda);
        if (t(i,0) != s){
          double delta_prime=(eta*delta_prior)/(1+eta);
          loglik=loglik+delta_prime*log(lambda)-delta_prime*log(1-lambda);
          score_t(s-1,0)=score_t(s-1,0)+delta_prime;
          score_v=score_v+delta_prime*Z.row(i).transpose();
        }
        if (t(i,0) == s){
          double delta_prime=(delta(i,0)+(eta*delta_prior))/(1+eta);
          loglik=loglik+delta_prime*log(lambda)-delta_prime*log(1-lambda);
          score_t(s-1,0)=score_t(s-1,0)+delta_prime;
          score_v=score_v+delta_prime*Z.row(i).transpose();
        }

    }
  }
  
  List result;
  result["loglik"]=loglik;
  result["score_t"]=score_t;
  result["score_v"]=score_v;
  result["info_t"]=info_t;
   
  return result;
}


// [[Rcpp::export]]
List NR_boostingKL_logit(Eigen::MatrixXd t, Eigen::MatrixXd z, Eigen::MatrixXd delta, Eigen::MatrixXd beta_t_prior, Eigen::MatrixXd LP_prior, Eigen::MatrixXd beta_t, Eigen::MatrixXd beta_z, double eta, double tol, int Mstop, double step_size){
  int m = 0;
  int P = z.cols();
  int n = z.rows();
  int K=t.maxCoeff();
  Eigen::MatrixXd beta2_t=beta_t; //should be a vec of mean delta[t]
  Eigen::MatrixXd beta2_z=beta_z; //should be a vec of 0
  double diff = R_PosInf;

  Eigen::MatrixXd A_inv(K,K);
  A_inv.setZero(K,K);
  Eigen::MatrixXd A(K,1);
  A.setZero(K,1);
  
  Eigen::MatrixXd dbeta_t(K,1);
  dbeta_t.setZero(K,1);
  Eigen::MatrixXd dbeta_z(P,1);
  dbeta_z.setZero(P,1);
  //Eigen::MatrixXd beta_z_matrix(2, Mstop+1); 
  //beta_z_matrix.setZero(2, Mstop+1);
  Eigen::MatrixXd loglik_vec(1, Mstop+1); 
  loglik_vec.setZero(1, Mstop+1);
  double ll0,ll,ll2;
  Eigen::MatrixXf::Index maxRow, maxCol;
  float max;
  List result;

  while (m<=Mstop){
    m = m+1;
    List update = dbetaKL_logit(t, beta_t_prior, LP_prior, z, delta, beta_t, beta_z, K, P, n, eta);
    dbeta_z = update["score_v"];
    if (m==1){
      ll0 = update["loglik"];
    }
    ll = update["loglik"];
    ll2 = ll;

    //update beta_Z by boosting
    max = (dbeta_z.array().abs()).matrix().maxCoeff(&maxRow, &maxCol);

    beta2_z(maxRow,maxCol)=beta2_z(maxRow,maxCol)+step_size*((dbeta_z.array().sign()).matrix()(maxRow,maxCol));
    //beta_z_matrix(0,m) = maxRow;
    //beta_z_matrix(1,m) = step_size*((dbeta_z.array().sign()).matrix()(maxRow,maxCol));

    //update beta_t by NR
    update = dbetaKL_logit(t, beta_t_prior, LP_prior, z, delta, beta_t, beta2_z, K, P, n, eta);
    A=update["info_t"];
    dbeta_t = update["score_t"];
    A=A.array().inverse();
    A_inv=A.asDiagonal();
    beta2_t=beta_t+(A_inv*dbeta_t);

    update = dbetaKL_logit(t, beta_t_prior, LP_prior, z, delta, beta2_t, beta2_z, K, P, n, eta);
    ll2 = update["loglik"];
    diff = abs(ll2-ll)/abs(ll2-ll0);
    beta_z = beta2_z;
    beta_t = beta2_t;
    loglik_vec(0,m) = ll2;

    result["beta_t"]=beta2_t;
    result["beta_v"]=beta2_z; 
    result["m"]=m;
    result["ll_path"]=loglik_vec;
    //result["path"]=beta_z_matrix;

    if (diff<tol) {
     return result;
    } 
  }
    return result;

}


// [[Rcpp::export]]
List dbeta_logit(Eigen::MatrixXd t, Eigen::MatrixXd Z, Eigen::MatrixXd delta, Eigen::MatrixXd beta_t, Eigen::MatrixXd beta_v, int K, int p, int n){
  double loglik=0;
  Eigen::MatrixXd score_t(K,1);
  score_t.setZero(K,1);
  Eigen::MatrixXd score_v(p,1);
  score_v.setZero(p,1);
  Eigen::MatrixXd info_t(K,1);
  info_t.setZero(K,1);
  double delta_prime=0;
  
  for (int i = 0 ; i < n ; i++){
    for (int s = 1 ; s <= t(i,0) ; s++){
        double lambda=1/(1+exp(-beta_t(s-1,0)-(Z.row(i)*beta_v)(0,0)));
        loglik=loglik+log(1-lambda);
        score_t(s-1,0)=score_t(s-1,0)-lambda;
        score_v=score_v-lambda*Z.row(i).transpose();
        info_t(s-1,0)=info_t(s-1,0)+lambda*(1-lambda);
        if (t(i,0) != s){
          delta_prime=0;
        }
        if (t(i,0) == s){
          delta_prime=delta(i,0);
        }
        loglik=loglik+delta_prime*log(lambda)-delta_prime*log(1-lambda);
        score_t(s-1,0)=score_t(s-1,0)+delta_prime;
        score_v=score_v+delta_prime*Z.row(i).transpose();
    }
  }
  
  List result;
  result["loglik"]=loglik;
  result["score_t"]=score_t;
  result["score_v"]=score_v;
  result["info_t"]=info_t;
   
  return result;
}


// [[Rcpp::export]]
List boosting_logit(Eigen::MatrixXd t, Eigen::MatrixXd z, Eigen::MatrixXd delta, Eigen::MatrixXd beta_t, Eigen::MatrixXd beta_z, double tol, int Mstop, double step_size){
  int m = 0;
  int P = z.cols();
  int n = z.rows();
  int K=t.maxCoeff();
  Eigen::MatrixXd beta2_t=beta_t; //should be a vec of mean delta[t]
  Eigen::MatrixXd beta2_z=beta_z; //should be a vec of 0
  double diff = R_PosInf;

  Eigen::MatrixXd A_inv(K,K);
  A_inv.setZero(K,K);
  Eigen::MatrixXd A(K,1);
  A.setZero(K,1);
  
  Eigen::MatrixXd dbeta_t(K,1);
  dbeta_t.setZero(K,1);
  Eigen::MatrixXd dbeta_z(P,1);
  dbeta_z.setZero(P,1);
  //Eigen::MatrixXd beta_z_matrix(2, Mstop+1); 
  //beta_z_matrix.setZero(2, Mstop+1);
  Eigen::MatrixXd loglik_vec(1, Mstop+1); 
  loglik_vec.setZero(1, Mstop+1);
  double ll0,ll,ll2;
  Eigen::MatrixXf::Index maxRow, maxCol;
  float max;
  List result;

  while (m<=Mstop){
    m = m+1;
    List update = dbeta_logit(t, z, delta, beta_t, beta_z, K, P, n);
    dbeta_z = update["score_v"];
    if (m==1){
      ll0 = update["loglik"];
    }
    ll = update["loglik"];
    ll2 = ll;

    //update beta_Z by boosting
    max = (dbeta_z.array().abs()).matrix().maxCoeff(&maxRow, &maxCol);

    beta2_z(maxRow,maxCol)=beta2_z(maxRow,maxCol)+step_size*((dbeta_z.array().sign()).matrix()(maxRow,maxCol));
    //beta_z_matrix(0,m) = maxRow;
    //beta_z_matrix(1,m) = step_size*((dbeta_z.array().sign()).matrix()(maxRow,maxCol));

    //update beta_t by NR
    update = dbeta_logit(t, z, delta, beta_t, beta2_z, K, P, n);
    A=update["info_t"];
    dbeta_t = update["score_t"];
    A=A.array().inverse();
    A_inv=A.asDiagonal();
    beta2_t=beta_t+(A_inv*dbeta_t);

    update = dbeta_logit(t, z, delta, beta2_t, beta2_z, K, P, n);
    ll2 = update["loglik"];
    diff = abs(ll2-ll)/abs(ll2-ll0);
    beta_z = beta2_z;
    beta_t = beta2_t;
    loglik_vec(0,m) = ll2;

    result["beta_t"]=beta2_t;
    result["beta_v"]=beta2_z; 
    result["m"]=m;
    result["ll_path"]=loglik_vec;
    //result["path"]=beta_z_matrix;

    if (diff<tol) {
     return result;
    } 
  }
    return result;
}



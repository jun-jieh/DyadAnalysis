// Input data
data {
  int<lower=0> N;      //number of observations
  int<lower=0> Nanim;  //number of animals (giver/receiver)
  int<lower=0> Ngroup; //number of social groups
  real y[N];           //duration of attacking time
  int receiver_id[N];  //receiver id from 1...Nanim
  int giver_id[N];     //giver id from 1...Nanim
  int group_id[N];     //social group id from 1...Ngroup
  vector[N] weight_receiver; //(centered) weight of receiver
  vector[N] weight_giver;    //(centered) weight of giver
  vector[N] sex;             //sex: 1= barrow
  vector[N] nursery;         //nursery mate status: 1=shared nursery pen
  vector[N] litter;          //litter mate status: 1=littermates
  
}

// The (co)variance matrix for the random giver effect and random receiver effect
transformed data{
  vector[2] mu;
  cov_matrix[2] S = [[5, 0], [0, 5]];
  
  for(i in 1:2) mu[i]=0; 
  
}

// The parameters accepted by the model.
parameters {
  real alpha;                 //intercept, lognormal
  real sex_eff;               //sex effect, lognormal
  real nursery_eff;           // common nursery effect, lognormal
  real litter_eff;            // common litter effect, lognormal
  real weight_receiver_eff;   //slope of receiver weight, lognormal
  real weight_giver_eff;      // slope of giver weight, lognormal
  
  vector[Ngroup] group_eff;   //group random effect, lognormal
  
  matrix[Nanim,2] Animal;     // matrix for the two animals, lognormal
  cov_matrix[2] Sigma;
  
  real<lower=0> sigma_e;      //residual standard deviation, lognormal
  real<lower=0> sigma_group;  //group standard deviation, lognormal

  real alpha0;                //intercept, Bernoulli
  real sex_eff0;              //sex effect, Bernoulli
  real nursery_eff0;          // common nursery effect, Bernoulli
  real litter_eff0;           // common litter effect, Bernoulli
  real weight_receiver_eff0;  //slope of receiver weight, Bernoulli
  real weight_giver_eff0;     // slope of giver weight, Bernoulli
  
  vector[Ngroup] group_eff0;  //group random effect, Bernoulli
  
  matrix[Nanim,2] Animal0;    // matrix for the two animals, Bernoulli
  cov_matrix[2] Sigma0;

  real<lower=0> sigma_group0; //group standard deviation, Bernoulli

}

transformed parameters {
  vector[N] eta;   //for each observation compute the linear predictor, lognormal
  vector[N] eta0;  //for each observation compute the linear predictor, Bernoulli
  real<lower=0, upper=1> theta[N]; //probability of being 0

  for (i in 1:N)
  eta[i]=alpha+sex[i]*sex_eff+litter[i]*litter_eff+nursery[i]*nursery_eff+weight_receiver[i]*weight_receiver_eff+weight_giver[i]*weight_giver_eff+group_eff[group_id[i]]+Animal[giver_id[i],1]+Animal[receiver_id[i],2];

  for (i in 1:N)
  eta0[i]=alpha0+sex[i]*sex_eff0+litter[i]*litter_eff0+nursery[i]*nursery_eff0+weight_receiver[i]*weight_receiver_eff0+weight_giver[i]*weight_giver_eff0+group_eff0[group_id[i]]+Animal0[giver_id[i],1]+Animal0[receiver_id[i],2];

  for (i in 1:N)
  theta[i] = inv_logit(eta0[i]);

}

model {
  Sigma ~ wishart(4, S);  
  Sigma0 ~ wishart(4, S);  

  
  sigma_group~uniform(0,10);                   //prior random eff
  sigma_e~uniform(0,100);                      //prior random eff

  group_eff~normal(0,sigma_group);             //prior random eff
  
  sex_eff~uniform(-10,10);                     //prior fixed eff
  litter_eff~uniform(-10,10);                  //prior fixed eff
  nursery_eff~uniform(-10,10);                 //prior fixed eff
  alpha~uniform(-10,10);                       //prior fixed eff
  
  weight_receiver_eff~uniform(-100,100);       //prior fixed eff
  weight_giver_eff~uniform(-100,100);          //prior fixed eff

  sigma_group0~uniform(0,10);                  //prior random eff
  group_eff0~normal(0,sigma_group0);           //prior random eff
  
  sex_eff0~uniform(-10,10);                    //prior fixed eff
  litter_eff0~uniform(-10,10);                 //prior fixed eff
  nursery_eff0~uniform(-10,10);                //prior fixed eff
  alpha~uniform(-10,10);                       //prior fixed eff
  
  weight_receiver_eff0~uniform(-100,100);       //prior fixed eff
  weight_giver_eff0~uniform(-100,100);          //prior fixed eff

  
  for (i in 1:Nanim)
      Animal[i]~multi_normal(mu,Sigma);
  for (i in 1:Nanim)
      Animal0[i]~multi_normal(mu,Sigma0);
  
  
  // Likelihood
  for (i in 1:N) {
    if (y[i] == 0) {
    target += log(theta[i]);
    } else {
      target += log1m(theta[i]) + lognormal_lpdf(y[i] | eta[i],sigma_e);
    }
  }


}


generated quantities{
  real <lower=0> sigma_receiver;
  real <lower=0> sigma_receiver0;
  real <lower=0> sigma_giver;
  real <lower=0> sigma_giver0;
  real <lower=-1, upper=1> cor_g_r;
  real <lower=-1, upper=1> cor_g_r0;
  real weight_diff;
  real weight_diff0;
  real y_rep[N];
  int struc_zero;
  
  sigma_receiver = Sigma[2,2];
  sigma_receiver0 = Sigma0[2,2];
  sigma_giver = Sigma[1,1];
  sigma_giver0 = Sigma0[1,1];
  cor_g_r = Sigma[1,2]/sqrt(sigma_giver*sigma_receiver);
  cor_g_r0 = Sigma0[1,2]/sqrt(sigma_giver0*sigma_receiver0);
  weight_diff = weight_giver_eff-weight_receiver_eff;
  weight_diff0 = weight_giver_eff0-weight_receiver_eff0;
  
  for (i in 1:N){
   struc_zero = bernoulli_rng(theta[i]);
   if (struc_zero == 1) {
     y_rep[i]=0.0;
   } else {
     y_rep[i]=lognormal_rng(eta[i],sigma_e);
   }

  }
  
}


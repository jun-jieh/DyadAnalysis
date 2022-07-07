// Input data
data {
  int<lower=0> N;   
  int<lower=0> Nanim; 
  int<lower=0> Ngroup; 
  real y[N]; 
  int receiver_id[N]; 
  int giver_id[N];    
  int group_id[N];    
  vector[N] weight_receiver; 
  vector[N] weight_giver;    
  vector[N] sex;             
  vector[N] nursery;         
  vector[N] litter;         
}



parameters {
  real alpha;                 //intercept
  real sex_eff;               //sex effect
  real nursery_eff;           // common nursery effect
  real litter_eff;            // common litter effect
  real weight_receiver_eff;   //slope of receiver weight
  real weight_giver_eff;      // slope of giver 
  
  vector[Ngroup] group_eff;   //group random effect
  vector[Nanim] giver_eff;    //giver random effect
  vector[Nanim] receiver_eff; //receiver random effect
  
  real<lower=0> sigma;            //residual standard deviation
  real<lower=0> sigma_giver;      //giver standard deviation
  real<lower=0> sigma_receiver;   //receiver standard deviation
  real<lower=0> sigma_group;      //group standard deviation
  real<lower=0, upper=1> theta;
}

transformed parameters {
  vector[N] eta;  //for each observation compute the linear predictor

  for (i in 1:N)
  eta[i]=alpha+sex[i]*sex_eff+litter[i]*litter_eff+nursery[i]*nursery_eff+weight_receiver[i]*weight_receiver_eff+weight_giver[i]*weight_giver_eff+group_eff[group_id[i]]+giver_eff[giver_id[i]]+receiver_eff[receiver_id[i]];

}

model {
  sigma_group~uniform(0,10);
  sigma~uniform(0,100);                     
  sigma_giver~uniform(0,10); 
  sigma_receiver~uniform(0,10);
  
  giver_eff~normal(0,sigma_giver);             
  receiver_eff~normal(0,sigma_receiver);       
  group_eff~normal(0,sigma_group);             
  
  sex_eff~uniform(-10,10);                     
  litter_eff~uniform(-10,10);                  
  nursery_eff~uniform(-10,10);                 
  alpha~uniform(-10,10);                       
  
  weight_receiver_eff~uniform(-100,100);       
  weight_giver_eff~uniform(-100,100);          
  
  // Likelihood
  for (i in 1:N) {
    if (y[i] == 0) {
    target += log_sum_exp(bernoulli_lpmf(1 | theta),
                            bernoulli_lpmf(0 | theta)
                              + normal_lpdf(y[i] | eta[i],sigma));
    } else {
      target += bernoulli_lpmf(0 | theta)
                  + normal_lpdf(y[i] | eta[i],sigma);
    }
  }


}

generated quantities {
  vector[N] y_rep;
  //int<lower=0> y_rep[N];
  for (i in 1:N){
      y_rep[i] = (1-bernoulli_rng(theta))*normal_rng(eta[i],sigma); 
  }
}


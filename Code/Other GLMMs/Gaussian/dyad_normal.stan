// Input data
data {
  int<lower=0> N;    //number of observations
  int<lower=0> Nanim; //number of animals (giver/receiver)
  int<lower=0> Ngroup; //number of social groups
  real y[N]; //counts
  int receiver_id[N]; //receiver id from 1...Nanim
  int giver_id[N];    //giver id from 1...Nanim
  int group_id[N];    //social group id from 1...Ngroup
  vector[N] weight_receiver; //weight
  vector[N] weight_giver;    //weight
  vector[N] sex;             //sex: 1= barrow
  vector[N] nursery;         //nursery mate status: 1=shared pen
  vector[N] litter;          //litermate status: 1=littermates
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
  
  real<lower=0> sigma;            //error standard deviation
  real<lower=0> sigma_giver;      //giver standard deviation
  real<lower=0> sigma_receiver;   //receiver standard deviation
  real<lower=0> sigma_group;      //group standard deviation
}

transformed parameters {
  vector[N] eta;  //for each observation compute the linear predictor

  for (i in 1:N)
  eta[i]=alpha+sex[i]*sex_eff+litter[i]*litter_eff+nursery[i]*nursery_eff+weight_receiver[i]*weight_receiver_eff+weight_giver[i]*weight_giver_eff+group_eff[group_id[i]]+giver_eff[giver_id[i]]+receiver_eff[receiver_id[i]];

}

model {
  sigma_group~uniform(0,10);                    
  sigma_giver~uniform(0,10); 
  sigma_receiver~uniform(0,10);
  sigma~uniform(0,100); 
  
  giver_eff~normal(0,sigma_giver);             
  receiver_eff~normal(0,sigma_receiver);       
  group_eff~normal(0,sigma_group);            
  
  sex_eff~uniform(-10,10);                     
  litter_eff~uniform(-10,10);                  
  nursery_eff~uniform(-10,10);               
  alpha~uniform(-10,10);                       
  
  weight_receiver_eff~uniform(-100,100);      
  weight_giver_eff~uniform(-100,100);         

  y ~ normal(eta,sigma);   //likelihood
}

generated quantities{
  real y_rep[N];
  
  for (i in 1:N){
        y_rep[i]=normal_rng(eta[i],sigma); 
    }
}


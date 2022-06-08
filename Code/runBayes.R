# Load all the required packages for the implementation
# Note: if the packages are not installed in the local computer, use install.packages() to install them
library(ggplot2)
library(bayesplot)
library(rstan)

# For execution on a local, multicore CPU with excess RAM the following two lines are recommended: 
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# Load the data file that is also provided in this GitHub repository
load("~/dyad_data.RData")
# Set working path (optional)
# setwd("~/Dyadic data analysis")

# Create a dataset for the stan program
# Response variable
y<-dyad_data$y
# Indicator of the previous litter mate experience 
litter<-as.numeric(dyad_data$litter)-1
# Sex of the dyad (pigs were housed in single-sex pens)
sex<-1*(dyad_data$sex=="b")
# Indicator of the previous nursery mate experience
nursery<-as.numeric(dyad_data$nursery)-1
# Social group index
social_group<-as.numeric(droplevels(dyad_data$social_group))
# Aggression giver ID
giver_id<-as.numeric(droplevels(dyad_data$giver_id))
# Aggression receiver ID
receiver_id<-as.numeric(droplevels(dyad_data$receiver_id))


# Put data into a list. Note: c_weight means the centered weight within social groups
data<-list(N=length(y),
           Nanim=length(unique(receiver_id)),
           Ngroup=length(unique(social_group)),
           y=y,
           receiver_id=receiver_id,
           giver_id=giver_id,
           group_id=social_group,
           weight_receiver=dyad_data$c_weight_receiver,
           weight_giver=dyad_data$c_weight_giver,
           sex=sex,
           nursery=nursery,
           litter=litter)

# Time profiler (start recording time)
strt_time <- Sys.time()

# Call the stan code generator, compile, and execute the model fitting
hurdle_model  <-  stan(file = "hurdle.stan",data = data,
                                      chains = 4,warmup = 5000,iter = 15000,thin = 10,seed=22222,
                                      control = list(max_treedepth = 12),
                                      pars = c("alpha",
                                               "sex_eff",
                                               "nursery_eff",
                                               "litter_eff",
                                               "weight_receiver_eff",
                                               "weight_giver_eff",
                                               "Sigma",
                                               "sigma_e",
                                               "sigma_group",
                                               "alpha0",
                                               "sex_eff0",
                                               "nursery_eff0",
                                               "litter_eff0",
                                               "weight_receiver_eff0",
                                               "weight_giver_eff0",
                                               "Sigma0",
                                               "sigma_group0",
                                               "sigma_receiver",
                                               "sigma_receiver0",
                                               "sigma_giver",
                                               "sigma_giver0",
                                               "cor_g_r",
                                               "cor_g_r0",
                                               "weight_diff",
                                               "weight_diff0",
                                               "y_rep"
                                               ))

# Save the 
save(hurdle_model, file="DyadAnalysis_output.RData")


# Time profiler (compute the time elapsed)
end_time <- Sys.time()
runtime <- end_time - strt_time
print(runtime)

# Convergence diagnostics
check_hmc_diagnostics(hurdle_model)

# We have so many variables. So, we will generate multiple trace plots of the MCMC chains
# Note: the variable names are explained in the stan code
mcmc_trace(hurdle_model,pars=c("alpha", "sex_eff","nursery_eff","litter_eff","weight_receiver_eff","weight_giver_eff"))

mcmc_trace(hurdle_model,pars=c("sigma_e",
                               "Sigma[1,1]",
                               "Sigma[1,2]",
                               "Sigma[2,2]",
                               "sigma_group"))

mcmc_trace(hurdle_model,pars=c("alpha0", "sex_eff0","nursery_eff0","litter_eff0","weight_receiver_eff0","weight_giver_eff0"))

mcmc_trace(hurdle_model,pars=c("Sigma0[1,1]",
                               "Sigma0[1,2]",
                               "Sigma0[2,2]",
                               "sigma_group0"))


# We have so many variables. So, we will generate multiple autocorrelation plots
# Note: the variable names are explained in the stan code
mcmc_acf(hurdle_model,pars=c("alpha", "sex_eff","nursery_eff","litter_eff","weight_receiver_eff","weight_giver_eff"))

mcmc_acf(hurdle_model,pars=c("sigma_e",
                             "Sigma[1,1]",
                             "Sigma[1,2]",
                             "Sigma[2,2]",
                             "sigma_group"))

mcmc_acf(hurdle_model,pars=c("alpha0", "sex_eff0","nursery_eff0","litter_eff0","weight_receiver_eff0","weight_giver_eff0"))

mcmc_acf(hurdle_model,pars=c("Sigma0[1,1]",
                             "Sigma0[1,2]",
                             "Sigma0[2,2]",
                             "sigma_group0"))

# Finally, obtain the posterior distribution of the estimated parameters for Bayesian inference
# We do not recommend running this line if the simulated data sets y_reps are obtained from the generated quantaties
round(summary(hurdle_model)$summary,3)

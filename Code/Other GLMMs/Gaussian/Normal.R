# Load all the required packages for the implementation
# Note: if the packages are not installed in the local computer, use install.packages() to install them
library(ggplot2)
library(bayesplot)
library(rstan)

# For execution on a local, multicore CPU with excess RAM the following two lines are recommended:
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores(logical = FALSE))

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

strt_time <- Sys.time()


# Call the stan code generator, compile, and execute the model fitting
gaussian_model  <-  stan(file = "dyad_normal.stan",data = data,
                       chains = 4,warmup = 1500,iter = 2500,thin = 4,seed=66666,
                       control = list(max_treedepth = 12),
                       pars = c("alpha",
                                "sex_eff",
                                "nursery_eff",
                                "litter_eff",
                                "weight_receiver_eff",
                                "weight_giver_eff",
                                "sigma_group",
                                "sigma_receiver",
                                "sigma_giver",
                                "sigma",
                                "y_rep"
                       ))

end_time <- Sys.time()
runtime <- end_time - strt_time
save(gaussian_model,runtime, file="Gaussian.RData")

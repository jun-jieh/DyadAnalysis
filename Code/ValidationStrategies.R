# Load all the required packages for the implementation
# Note: if the packages are not installed in the local computer, use install.packages() to install them
library(ggplot2)
library(bayesplot)
library(rstan)
library(loo)

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




############# stratified 5-fold (random) cross-validation #############

# K-fold crosss-validation. nf: the number of folds
nf <- 5
# Create K folds according to the social group
fold <- kfold_split_stratified(K = nf, x = dyad_data$social_group)
# Save replicated y from each fold of the testing sets
# Note: this number should match the number of generated y_reps from the stan program
# Check the stan function specification to match the number
y_rep_kfold <- matrix(nrow = 500, ncol = nrow(dyad_data))

# Time profiler
strt_time <- Sys.time()
# Loop over the nf folds
for(i in 1:nf){
  # Select random observations within each social group for testing purposes
  holdout <- fold==i
  # Put data into list
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
             litter=litter, 
             holdout=holdout)
  
  cat("Now the Fold", i,"is running \n")
  # Call stan code generator, compile, and execute
  kfold_cv  <-  stan(file = "hurdle_validation.stan",data = data,
                     chains = 4,warmup = 5000,iter = 15000,thin = 40,seed=33333,
                     control = list(max_treedepth = 12),
                     pars = c("y_rep"))
  
  # Only collect the predictions for the testing set
  # (drop the predictions for the training data)
  yrep_tmp <- extract(kfold_cv,"y_rep")$y_rep
  y_rep_kfold[, fold == i] <- yrep_tmp[, fold == i]
  # Save the results for the specific fold (optional)
  #filen<-paste("eff_cv",i,"s.RData",sep = "_")
  #save(kfold_cv,file = filen)
}

# Time profiler
end_time <- Sys.time()
runtime <- end_time - strt_time
print(runtime)
save(y_rep_kfold,kfold_cv, file="RandomCV.RData")



############# Block-by-social-group cross-validation #############
nf <- 5
# Create K folds according to the social group
fold <- kfold_split_grouped(K = nf, x = dyad_data$social_group)
# Save replicated y from each fold of the testing sets
# Note: this number should match the number of generated y_reps from the stan program
# Check the stan function specification to match the number
y_rep_groupCV <- matrix(nrow = 500, ncol = nrow(dyad_data))

# Time profiler
strt_time <- Sys.time()
# Loop over the folds
for(i in 1:nf){
  # Select random observations by social group for testing purposes
  holdout <- fold==i
  #put data into list
  data <- list(N=length(y),
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
               litter=litter, 
               holdout=holdout)
  
  cat("Now the Fold", i,"is running \n")
  
  # Call stan code generator, compile, and execute
  group_cv <- stan(file = "hurdle_validation.stan",data = data,
                   chains = 4,warmup = 5000,iter = 15000,thin = 40,seed=44444,
                   control = list(max_treedepth = 12),
                   pars = c("y_rep"))
  
  yrep_tmp <- extract(group_cv,"y_rep")$y_rep
  y_rep_groupCV[,fold == i] <- yrep_tmp[,fold == i]
  # Save the results for the specific fold (optional)
  #filen<-paste("gr_eff_cv",i,"s.RData",sep = "_")
  #save(tr_cv,file = filen)
}


# Time profiler
end_time <- Sys.time()
runtime <- end_time - strt_time
print(runtime)
save(fold, y_rep_groupCV, models, file="GroupCV.RData")

############# Block-by-focal-animals validation #############
nrep = 5

# select all observations from seven random animal per group.
anim<-as.character(unique(dyad_data$receiver_id))

# position of each animal
posi <- match(anim,as.character(dyad_data$receiver_id))

# get social group info
sg <- as.character(dyad_data$social_group[posi])

# Save the indices of observations reserved for testing purposes in each replicate
index_test <- matrix(nrow = nrep, ncol = nrow(dyad_data))

# Save simulated y from each replicate
y_rep_focal <- NULL

# Loop over the folds
for(i in 1:nrep){
  
  # Select seven focal animals from each social group
  # The remaining non-focal animals are reserved for testing
  animal_train <- as.vector(by(data = anim,INDICES = sg,FUN = sample,size=7))
  animal_train <- as.character(unlist(animal_train))
  
  # Get the indices for the training set (records of focal animals)
  tmp_train <- (as.character(dyad_data$giver_id)%in%animal_train)|(as.character(dyad_data$receiver_id)%in%animal_train)
  # The non-focal animals are used for testing purposes (holdout)
  holdout <- !tmp_train
  # Save the holdout indices
  index_test[i,] <- holdout
  
  #put data into list
  data <- list(N=length(y),
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
               litter=litter, 
               holdout=holdout)
  
  cat("Now the Fold", i,"is running \n")
  
  # Call stan code generator, compiler, and excution
  focal_cv  <-  stan(file = "hurdle_validation.stan",data = data,
                     chains = 4,warmup = 5000,iter = 15000,thin = 40,seed=55555,
                     control = list(max_treedepth = 12),
                     pars = c("y_rep"))
  
  yrep_tmp <- extract(focal_cv,"y_rep")$y_rep
  y_rep_focal[[i]] <- yrep_tmp
  save(index_test, y_rep_focal, file="FocalAnimalCV.RData")
}


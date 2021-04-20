
install.packages("gtools")
library(gtools)

x<- c(5,10,15)
perm <- permutations(length(x),2,x,repeats.allowed = TRUE)
perm


mean_perm <- rowMeans(perm)
mean_perm


mean(rowMeans(perm)) == mean(x)


hist(mean_perm,probability =TRUE)



par(mfrow=c(1,2))
hist(rpois(n=100000, lambda = 20), freq = F)
hist(rnorm(n=100000, mean = 20 , sd= 20^0.5), freq = F)


par(mfrow=c(1,2))
hist(rbinom(n=100000, size = 20, prob = 0.4), freq = F)
hist((rnorm(n=100000,mean = 20*0.4, sd=(20*0.4*(1-0.4))^0.5)), ferq=F))



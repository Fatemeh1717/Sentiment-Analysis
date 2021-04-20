
#1

library(MASS)
attach(cats)

str(cats)

plot(cats$Bwt,cats$Hwt)

cor(cats$Bwt,cats$Hwt, method = "pearson")


x<-cats$Bwt
y<-cats$Hwt
r <- sum((x-mean(x))* (y-mean(y)))/ ((length(x)-1)*sd(x)*sd(y))
r
# There is a strong positive linear relationship betwwen x, y

cov(x,y , method = "pearson")
correlation <- cov(x,y , method = "pearson")/ (sd(x)*sd(y))
correlation

corrmethod<-cov(x,y)*sd(x)/sd(y)
corrmethod  
cor(cats$Bwt,cats$Hwt)

slmodel <- lm(y~x , data = cats)
summary(slmodel)

coef(slmodel)[2]

#2

data(mtcars)
summary(mtcars)


var <- c("hp","cyl","mpg","gear")

pairs(mtcars[var])


cor(mtcars$cyl,mtcars$hp , method = "pearson")
cor(mtcars$mpg,mtcars$hp , method="spearman")


table_cor_mtcars <- cor(mtcars)

install.packages("corrplot")
library(corrplot)
corrplot(table_cor_mtcars, method = "ellipse")
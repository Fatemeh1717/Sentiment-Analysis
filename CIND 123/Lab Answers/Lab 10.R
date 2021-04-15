
#1

library(MASS)
attach(cats)

str(cats)

#2
plot(cats$Bwt,cats$Hwt)

#3
cor.test(cats$Bwt,cats$Hwt, method = "pearson")

#4
x<-cats$Bwt
y<-cats$Hwt
r <- sum((x-mean(x))* (y-mean(y)))/ ((length(x)-1)*sd(x)*sd(y))
r

#5
# There is a strong positive linear relationship betwwen x, y

 
#7
cov(x,y , method = "pearson")
correlation <- cov(x,y , method = "pearson")/ (sd(x)*sd(y))
correlation


corrmethod<-cov(x,y)*sd(x)/sd(y)
corrmethod  
cor(cats$Bwt,cats$Hwt)

#8
slmodel <- lm(y~x , data = cats)
summary(slmodel)

coef(slmodel)[2]

cor(cats$Bwt,cats$Hwt)* sd(cats$Hwt)/sd(cats$Bwt)

#2
#2-1
data(mtcars)
summary(mtcars)

#2-2
var <- c("hp","cyl","mpg","gear")

pairs(mtcars[var])

#2-3
cor(mtcars$cyl,mtcars$hp , method = "pearson")
cor(mtcars$mpg,mtcars$hp , method="spearman")


#2-4
cor(mtcars$mpg,mtcars$hp , method = "pearson")
cor(mtcars$mpg,mtcars$hp,method = "spearman")

-------------------------------------------------------------------------------
#corrplot
  
table_cor_mtcars <- cor(mtcars)

install.packages("corrplot")
library(corrplot)
corrplot(table_cor_mtcars, method = "ellipse")
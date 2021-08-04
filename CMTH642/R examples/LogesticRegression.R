# Logistic Regression example on am~wt on Mtcaras Dataset

str(mtcars)
model <- glm(formula = am ~ wt , data = mtcars, family = "binomial")
model
range(mtcars$wt)
x<- seq(min(mtcars$wt), max(mtcars$wt), 0.1)
x
y <- predict(model,list(wt=x), type = "response")
y
plot(mtcars$wt, mtcars$am)
lines(x,y)
summary(model)


x<- c(1,0,2,0,3,0,100)
sd(x)
y <-sum(x)
y
z <-mean(x)
z

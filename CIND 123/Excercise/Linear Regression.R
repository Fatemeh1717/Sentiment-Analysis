# Linear Regression in R

library(ISwR)
data(thuesen)

head(thuesen)
summary(thuesen)
str(thuesen)

# Scatterplot for the data
plot(thuesen)


# Are there any missing value? How Many and in which column?
sum(is.na(thuesen$short.velocity))
which(is.na(thuesen$short.velocity))

sum(is.na(thuesen$blood.glucose))

# Remove the missing values

new_thuesen <- na.omit(thuesen)
lm(short.velocity~blood.glucose, data=new_thuesen)
new_blood_glucose <-14
new_short_velocity <- 1.09781+0.02196*new_blood_glucose
new_short_velocity


cor(thuesen$blood.glucose, thuesen$short.velocity , use = "complete.obs")
cor(thuesen[,1:2], use = "complete.obs")




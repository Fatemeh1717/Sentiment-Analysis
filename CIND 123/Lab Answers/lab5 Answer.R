#2-1) Read and Store the online dataset using the read.csv() function

url <- "http://archive.ics.uci.edu/ml/
machine-learning-databases/auto-mpg/auto-mpg.data"

cars <- read.csv(file=url, stringsAsFactors = F, sep = "",
                 header = F)


names(cars)<- c('mpg', 'cyl', 'displacement', 'hp', 'weight',
                'acc', 'year', 'origin', 'name')



#2-2) Use summary() and str() functions to summarize the statistics of the attributes of the
#cars dataframe structure.

summary(cars)

str(cars)

head(cars)  

#2-3) Probably you noticed that the type of the horsepower attribute, denoted by hp has been classified as character.
#Use the as.numeric() function to resolve this issue.


cars$hp <- as.numeric(cars$hp)

#2-4) Create a scatter plot using car weight variable weight in the x-axis and acceleration
#variable acc in the y-axis.

plot(cars$weight, cars$acc)


#2-5) Select the cars that have the median acceleration and find the lightest (in terms of weight) ones.

index <- which(cars$acc == median(cars$acc))
subindex <- which.min(cars$weight[index])
cars$name[index][subindex]


#2-6) Select the cars whose weight is closest to the average value of all cars in the dataset.

cars$wdiff <- cars$weight-mean(cars$weight)
index2 <- which.min(abs(cars$wdiff))
cars$name[index2]


#3 Write an R function that takes in a set of numbers (student marks) and determines the correspondent grades.

gradefun <- function(marks){
  
 for(i in 1:length(marks)){
   
   if (marks[i]<= 60 ){print("Grade=F")}
   else if (marks[i] <= 70){print("Grade=D")}
   else if(marks[i]<= 80){print("Grade=C")}
   else if (marks[i]<=90){print("Grade=B")}
   else if (marks[i]>90 ){print("Grade=A")}
   
 }  
  
}

gradefun(c(20,50,45,89,90,100))







# 2-1) Create a function that calculates the average of a set of numbers

meanfunction <- function(x){
  
  return(sum(x, na.rm = T)/length(na.omit(x)))
}

meanfunction(c(2,3,4,NA))

# 2-2) Create a function that calculates the relative standing measure z-score of a specific value.

zscore <- function(x,y){
  
  return((x-meanfunction(y))/sd(y))
  
}
 zscore(2,c(1,8,3))
 
 
# 3)  Download the given dataset lab4dataset.csv then use the read.csv() function
 
 lab4data <- read.csv("lab4dataset.csv",
                      sep=",",
                      stringsAsFactors = F,
                      na.strings = c("","NA"))
# 4) Use the str() function to see more information about each attribute.

 str(lab4data)

#5) Assuming the dataset represents house prices and clients are usually concern with the
#maximum price they are willing to spend on a house. With this maximum price, create
#a function that returns a vector with 2 elements: i) the number of available houses
#at/below that price, and ii) the ratio of the number of those houses to the total number
#of the houses in the given dataset.  

#### number of houses below price & ratio#   
 housedata <- function(maxprice){
   
   x= sum(lab4data$price <= maxprice)
   y= x/length(lab4data$price)
   return(c(x,y))
   
 }
housedata(96377) 


#6) Create a function that takes in a set of numbers and plots it in a histogram, and returns
#1 if the set is skewed to the right, and -1 if it is skewed to the left. 

#### plots and determines skewness#
Skewness <- function(x)
{
   hist(x, freq = F, breaks = 50)
   return(sign(mean(x) - median(x)))
}
Skewness(lab4data$price)


#7) Create a function that has price as an input, and returns the z-score of the
#average number of bath-pieces (denoted by bathp in the dataset) for houses less than
#or equal to the inputted price. Also make it display a summary for the number of bath
#pieces for houses at or below the inputted price.

#### returns z score, prints summary#

bath <- function(price){
   
   print(summary(lab4data$bathp[lab4data$price <= price]))
   
   return(zscore(mean(lab4data$bathp[lab4data$price<=price]), lab4data$bathp ))
   
}
bath(49120) 


#8) Create a function that takes in a higher price and lower price as two arguments and
#creates a subset of houses that falls into that interval, then returns the first six rows of
#that subset.

#### makes subset of houses between price#

test <- function(upper,lower){
   
   
   sub<-0
   
   sub <- lab4data[which(lab4data$price >= lower & lab4data$price <= upper), ]
   
   return(head(sub))
   
}
 test(49120,25000)   


#9) Create a function that has an input as the number of bath-pieces, and returns a subset of the 
#cheapest house(s) that have that number of bath-pieces.
#### returns a subset of houses with x amount of pieces#

bathpieces <- function(pieces){
   
   bath <-0
   bath <- lab4data[which(lab4data$bathp == pieces), ] 
   bath <- bath[bath$price== min(bath$price),]
   return(bath)
   
} 

bathpieces(3) 
 
  

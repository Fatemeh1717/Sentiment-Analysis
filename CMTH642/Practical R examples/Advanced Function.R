#In this script we introduce the following function:

#Cut Function
#Sub function
#gsub” function
#split” function
#merge” function


#Cut Function divides the range of x into intervals and codes the values in x according to which interval they fall.
#Let’s create a factor in which the “blood.glucose” variable in the “thuesen” data set will be divided into four intervals
#(4, 7], (7, 9], (9, 12], and (12, 20]. The related code will be:
library("ISwR")
attach(thuesen)
str(thuesen)
int <- cut(blood.glucose,c(4,7,9,12,20))
levels(int) <- c("low","intermadte", "high","very high")
------------------------------------------------------------------------------------------------------------------------------------------
#The “sub” function replaces the first match of a string, if the parameter is a string vector, replaces the first match of all elements.
#Let's use the "sub" function and replace the "Data Analytics: Basic Methods" to "Data Analytics: Advanced Methods".

x<- "data analytics: basic metod"
y<- sub("basic","advanced",x)
y
----------------------------------------------------------------------------------------------------------------------------------------------
#The “gsub” function replaces all matches of a string, if the parameter is a string vector, returns a string vector of the same length and
#with the same attributes. Elements of string vectors which are not substituted will be returned unchanged.

x <- c("CIND 123: Spring Term CMTH: Spring Term")
gsub("Spring","Summer",x)

----------------------------------------------------------------------------------------------------------------------------------------------
# the “split” function divides the data in the vector x into the groups defined by f. The replacement forms replace values corresponding 
#to such a division. The “unsplit” function reverses the effect of split. Let's split the "energy" data set by the "stature" column. 
  
split(energy,energy$stature) 

#And now, let's expand the "expend" column of the "energy" dataset by the "stature" column.

split(energy$expend,energy$stature) 

----------------------------------------------------------------------------------------------------------------------------------------------------
  
#The “merge” function will merge the two data frames by common columns or row names.

#Let's create two data frames with one common column indicating "ID".
  
data.frame.A <- data.frame(ID = 1:4,
                           Gender = c("f","f","m","m"),
                           Age = c(30, 24, 26,18))

data.frame.B <- data.frame(ID = 1:4,
                           Weight = c(60,54,70,76),
                           Height = c(160,170,172,186))


total <- merge(data.frame.A, data.frame.B , by="ID")
total  
  
  
  




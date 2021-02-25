### Lab 2
#1)
FireName <- c("Waskesiu CFB","Birch Bay","Waskesiu CFB",
              "Wasstrom’s Flats","Millard","Rabbit","Sandy North",
              "Namekus Lake","Waskesiu CFB","Millard","National","Wasstrom’s Guards","South End Meadows")


BurnedArea <- c(40,0.1,NA,834,1483,20228,NA,1.2,56,693,0.5,30,830)
AverageBurned <- sum(BurnedArea, na.rm=TRUE)/11
AverageBurned

#2)
# 2-1
Year <- c("2019","2019","2018","2018","2018","2018","2018","2018","2017","2017","2017"
          ,"2017","2017","2016")


# 2-2
Fires <- cbind(Year, FireName,BurnedArea)
Fires

# 2- 3
# the type of all the variables has been changed to 'string'.

# 2-3
Fires[3,2]
Fires[6,]
dim(Fires)
ncol(Fires) 

# 3-1

matrix1 <- matrix(nrow = 3, ncol = 3)

# 3-2 
matrix1[1,] <- c(1,2,3)
matrix1[2,] <- c(4,2,1)
matrix1[3,] <- c(2,3,0)
matrix1

matrix1t <- t (matrix1)
matrix1t

# 3-3
matrixi <- solve(matrix1)
matrixi
 
  
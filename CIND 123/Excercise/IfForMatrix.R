# fill the matrix with nested for loop.


imatrix <- matrix(nrow=3 , ncol=4 )

imatrix[1,]<-c(4,5,2,1)
imatrix[2,]<-c(7,1,6,4)
imatrix[3,]<-c(8,3,2,6)
imatrix
imatrix <- c(1,2,3)
imatrix

# Define a matrix by usung nested for loop

a.mtrx <- matrix(nrow = 5, ncol = 5)
for (i in 1:5) {
  for (j in 1:5) {
    a.mtrx[i, j] <- abs(i - j)
  }
}

a.mtrx


b.mtrx <- matrix(nrow = 5, ncol = 5)
for (i in 1:5) {
  for (j in 1:5) {
    b.mtrx[i, j] <- i - j
  }
}

b.mtrx




c.mtrx <- matrix(nrow = i, ncol = j)
for (i in 1:5) {
  for (j in 1:5) {
   c.mtrx[i, j] <- i - j
  }
}
c.mtrx


d.mtrx <- matrix(nrow = 5, ncol = 5)
for (i in 1:5) {
  for (j in 1:5) {
    d.mtrx[i, j] <- i - 1
  }
}


d.mtrx


#--------------------------------------------------------------------

# Else- If 

#1) every part is ok

x<--1

if(x<0){
  print("negative")
} else if(x>0){
  print("positive")
}else
  print("zero")


#2) Syntax Error

if(x<0){
  print("negative")
}
else if(x>0){
  print("positive")
}
else(x==0){
  print("zero")
}


#3) Syntax Error

if(x<0){
  print("negative")
  else if (x>0){
    print("positive")
    else
      print("zero")
  }
    
}

#4) Just Display Negative Number

if(x<0){
  print("negative")
  if(x>0){
    print("positive")
    if(x==0){
      print("zero")
    }
  }
}

#------------------------------------------------------------
# Logical Operator

x <- T
y <- F
z <- NA

x&x
x|y
y&z


x&z
x&y
y|z


x&z
z|y
y|z


z&x
z|z
x&x




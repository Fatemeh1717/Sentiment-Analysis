
#2-1) Download the given dataset lab3dataset.csv then use the read.csv()


lab3data <- read.csv("lab3dataset.csv",
                     header = FALSE,
                     stringsAsFactors = FALSE,
                     na.strings = c("","NA"))

#2-2) Display the first 6 rows of the lab3dataset using the head() function

head(lab3data)

#2-3) Give proper names to each column using the names() function

names(lab3data) <- c('ID','Fname', 'Lname', 'Email', 'Gender',
                        'Country', 'Amount', 'Date')


#2-4) Use the str() function to see more information about each column.

str(lab3data)

#2-5) Find out how many different countries are there in the dataset by using the unique() function

length(unique(na.omit(lab3data$Country)))

#2-6) Count the number of Females under the Gender column

length(which(lab3data$Gender=='Female'))
sum(lab3data$Gender=='Female', na.rm = T)


#2-7) Count the number of NAs under the Gender column.

length(which(is.na(lab3data$Gender)))
sum(is.na(lab3data$Gender))


#3-1-a) Removing clients who have NA as their country.

lab3data <- lab3data[!is.na(lab3data$Country),]


#3-1-b) Converting the column with dollar values to numeric values.

lab3data$Amount <- as.numeric(gsub("[$,]","",lab3data$Amount))


#3-1-c) Converting the Date column to data type date.

lab3data$Date <- as.Date(lab3data$Date, "%m/%d/%Y")

head(lab3data)

#3-2) Identify the earliest date in the dataset and calculate the number of days passed for
# each observation. Insert these values as a new column to the data frame.


lab3data$Days <- as.numeric(lab3data$Date-min(lab3data$Date))
head(lab3data)

#3-3) Create an additional column then populate it with a numeric indicator where emails
#end with .gov, .org, or .net get 1 and the rest get 0.

lab3data$IndEmail <- 0


lab3data$IndEmail[grep(".gov",lab3data$Email)] <-1
lab3data$IndEmail[grep(".org", lab3data$Email)] <-1
lab3data$IndEmail[grep(".net",lab3data$Email)] <-1


head(lab3data)

#4) Check if the calculated number of days and the email indicator can be used to explain the amount column

lma <- lm(lab3data$Amount ~ lab3data$Days + lab3data$IndEmail)
summary(lma)








       
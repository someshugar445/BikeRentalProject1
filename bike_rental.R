# Load all the packages required for the analysis
library(ggplot2) # Visualisation
library(dplyr) # Data Wrangling
library(e1071) # Prediction: SVR
library(randomForest) # Prediction: Random Forest
library(corrplot)


# Remove all objects from R
rm(list = ls())

setwd("/home/someshugar/Project 1")

data = read.csv("/home/someshugar/Project 1/day.csv")

View(data)

str(data)

summary(data)

table(is.na(data))

data$season=as.factor(data$season)
data$weather=as.factor(data$weather)
data$holiday=as.factor(data$holiday)
data$workingday=as.factor(data$workingday)

plot(data$cnt~data$dteday,xlab="days", ylab="count of bikes rented")
     
boxplot(data$cnt~data$season,xlab="Seasons", ylab="count of users")
boxplot(data$cnt~data$mnth,xlab="months", ylab="count of users")
boxplot(data$cnt~data$holiday,xlab="Hoiday", ylab="count of users")
boxplot(data$cnt~data$weathersit,xlab="Weather", ylab="count of users")


plot(data$temp, data$cnt ,type = 'h',xlab = 'Actual Temperature', 
     ylab = 'Number of bikes rented')

plot(data$atemp, data$cnt ,type = 'h',xlab = 'Actual Feel Temperature', 
     ylab = 'Number of bikes rented')

plot(data$windspeed, data$cnt ,type = 'h',xlab = 'Actual Windspeed', ylab = 'Total Bike Rentals')

plot(data$hum, data$cnt ,type = 'h',xlab = 'Actual Humidity', ylab = 'Total Bike Rentals')


date<-substr(data$dteday,1,10)
days<-weekdays(as.Date(date))
data$day=days

boxplot(data$cnt~data$day,xlab="days", ylab="count of users")

data_cor<- data.frame(data %>% select( cnt,yr,weathersit,temp,atemp,hum,windspeed,casual,registered))
data_cor <- (cor(data_cor))
corrplot(data_cor)

# The features temp and atemp are strongly correlated. If both features are included in the model, 
# this will cause the issue of Multicollinearity (a given feature in the model can be approximated 
# by a linear combination of the other features in the model).
# 
# Hence I include only one temperature feature into the model.
# 
# The features casual and registered are omitted because that is what we are going to predict.

data$day=as.factor(data$day)
data$mnth=as.factor(data$mnth)

# Create a Simple Linear Regression model
train=data[1:547,]
View(train)
test=data[547:nrow(data), ]

lin.mod <- lm(cnt ~ season + holiday + workingday + weathersit + temp  + hum 
              + windspeed , data = train)
pred=predict(lin.mod,test)
test$logcnt=pred

c(adjusted.R.squared = summary(lin.mod)$adj.r.squared)
summary(lin.mod)



train_df <- data[1:547, ]
test_df <- data[547:nrow(data), ]

# Create a Random Forest model
rf_model <- randomForest(cnt ~ temp+ workingday + weathersit + atemp + hum + windspeed,
                         data = train_df)
print(rf_model)

# Predicting on test set
predTrain <- predict(rf_model, test_df)
View(predTrain)
# Visualizing the Random Forest Plot
plot(data$instant, data$cnt, type = "l", col = "red", xlab = "Day", ylab = "Number of Bike Users", main = "linear regression Plot for Bike Users")
legend("topleft", c("Actual", "Estimated"), lty = c(1, 1), col = c("red", "blue"))
lines(test_df$instant, predTrain, type = "l", col = "blue")

bikepred <- data.frame(count = as.integer(predTrain))

write.csv(bikepred
          ,file = "bikepred.csv"
          ,row.names = FALSE
          ,quote = FALSE
)





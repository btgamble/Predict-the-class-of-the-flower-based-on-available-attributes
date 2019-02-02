# https://machinelearningmastery.com/machine-learning-in-r-step-by-step/
# Examples in R 3.2
install.packages("caret", dependencies=c("Depends", "Suggests"))
# install.packages("caret")
# The downloaded source packages are in
# 'C:\Users\Travis\AppData\Local\Temp\Rtmp40vmns\downloaded_packages'

# install.packages("ggplot2", dependencies=c("Depends", "Suggests"))
# install.packages("pkgconfig", dependencies=c("Depends", "Suggests"))
# install.packages("ModelMetrics", dependencies=c("Depends", "Suggests"))
# install.packages("assertthat", dependencies=c("Depends", "Suggests"))
# install.packages("bindrcpp", dependencies=c("Depends", "Suggests"))
# install.packages("bindr", dependencies=c("Depends", "Suggests"))
# install.packages("generics", dependencies=c("Depends", "Suggests"))
# install.packages("gower", dependencies=c("Depends", "Suggests"))
# library(ggplot2)
# might have to explicitly call this after getting featurePlot to work
# install.packages("ellipse")

# load package 
library(caret)

# 2.1 Load Data The Easy Way
# attach the iris dataset to the environment
data(iris)
# rename the dataset
dataset <- iris

# 2.2 Load From CSV
# Maybe your a purist and you want to load the data just like you would on 
# your own machine learning project, from a CSV file.
# 
# Download the iris dataset from the UCI Machine Learning Repository 
# (here is the direct link).
# Save the file as iris.csv your project directory.
# Load the dataset from the CSV file as follows:

# define the filename
filename <- "iris.csv"
# load the CSV file from the local directory
dataset <- read.csv(filename, header=FALSE)
# set the column names in the dataset
colnames(dataset) <- c("Sepal.Length","Sepal.Width","Petal.Length","Petal.Width","Species")

# 2.3. Create a Validation Dataset

# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(dataset$Species, p=0.80, list=FALSE)
# select 20% of the data for validation
validation <- dataset[-validation_index,]
# use the remaining 80% of data to training and testing the models
dataset <- dataset[validation_index,]

# 3. Summarize Dataset
# 1.	Dimensions of the dataset.
# 2.	Types of the attributes.
# 3.	Peek at the data itself.
# 4.	Levels of the class attribute.
# 5.	Breakdown of the instances in each class.
# 6.	Statistical summary of all attributes.


# 3.1 Dimensions of Dataset
# how many instances (rows) and how many attributes (columns) 

# dimensions of dataset
dim(dataset)


# 3.2 Types of Attributes
# doubles, integers, strings, factors and other types.

# list types for each attribute
sapply(dataset, class)


#3.3 Peek at the Data

# take a peek at the first 5 rows of the data
head(dataset)


# 3.4 Levels of the Class

# list the levels for the class
levels(dataset$Species)

# multi-class or a multinomial classification problem. If there were two levels, 
# it would be a binary classification problem

# 3.5 Class Distribution

# summarize the class distribution
percentage <- prop.table(table(dataset$Species)) * 100
cbind(freq=table(dataset$Species), percentage=percentage)


# 3.6 Statistical Summary
# mean, the min and max values as well as some percentiles (25th, 50th or median and 75th 
                                                          
# summarize attribute distributions
summary(dataset)

# 4. Visualize Dataset
# 
# We are going to look at two types of plots:
# 1. Univariate plots to better understand each attribute.
# 2. Multivariate plots to better understand the relationships between attributes.
# 
# 4.1 Univariate Plots
#   - plots, that is, plots of each individual variable.

# split input and output
x <- dataset[,1:4]
y <- dataset[,5]

# boxplot for each attribute on one image
par(mfrow=c(1,4))
  for(i in 1:4) {
  boxplot(x[,i], main=names(iris)[i])
}
# shows distribution of the input attributes:

# barplot for class breakdown
plot(y)

# 4.2 Multivariate Plots
#   - look at the interactions between the variables.
# install.packages("featurePlot") # package 'featurePlot' is not available (for R version 3.4.3)
# library(featurePlot)
# scatterplot matrix
featurePlot(x=x, y=y, plot="ellipse") # could not find function "featurePlot"
# 
# box and whisker plots for each attribute
featurePlot(x=x, y=y, plot="box") # could not find function "featurePlot"

# distribution 
# density plots for each attribute by class value
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

# 5. Evaluate Some Algorithms
#  - create some models of the data and estimate their accuracy on unseen data.
# 
# Here is what we are going to cover in this step:
# 1. Set-up the test harness to use 10-fold cross validation.
# 2. Build 5 different models to predict species from flower measurements
# 3. Select the best model.
# 
# 5.1 Test Harness
# We will 10-fold crossvalidation to estimate accuracy.
# 
# This will split our dataset into 10 parts, train in 9 and test on 1 and release for 
# all combinations of train-test splits. 
# We will also repeat the process 3 times for each algorithm with different splits of the 
# data into 10 groups, in an effort to get a more accurate estimate.


# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# We are using the metric of "Accuracy" to evaluate models. This is a ratio of the number of 
# correctly predicted instances in divided by the total number of instances in the dataset 
# multiplied by 100 to give a percentage (e.g. 95% accurate). 
# 
# 5.2 Build Models
# 
# Will evaluate 5 different algorithms:
#   1. Linear Discriminant Analysis (LDA)
#   2. Classification and Regression Trees (CART).
#   3. k-Nearest Neighbors (kNN).
#   4. Support Vector Machines (SVM) with a linear kernel.
#   5. Random Forest (RF)
# 
# We reset the random number seed before reach run to ensure that the evaluation of each 
# algorithm is performed using exactly the same data splits. 
# It ensures the results are directly comparable.

# a) linear algorithms
set.seed(7)
fit.lda <- train(Species~., data=dataset, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(Species~., data=dataset, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(7)
fit.knn <- train(Species~., data=dataset, method="knn", metric=metric, trControl=control)
# c) advanced algorithms
# SVM
set.seed(7)
fit.svm <- train(Species~., data=dataset, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(7)
fit.rf <- train(Species~., data=dataset, method="rf", metric=metric, trControl=control)

# Caret does support the configuration and tuning of the configuration of each model, 
# but we are not going to cover that in this tutorial.

# 5.3 Select Best Model
# We now have 5 models and accuracy estimations for each. We need to compare the models to each 
# other and select the most accurate.

# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

# We can see the accuracy of each classifier and also other metrics like Kappa:

# We can also create a plot of the model evaluation results and compare the spread and the 
# mean accuracy of each model. There is a population of accuracy measures for each algorithm 
# because each algorithm was evaluated 10 times (10 fold cross validation).

# compare accuracy of models
dotplot(results)

#We can see that the most accurate model in this case was LDA:
  
#The results for just the LDA model can be summarized.

# summarize Best Model
print(fit.lda)

# This gives a nice summary of what was used to train the model and the mean and standard 
# deviation (SD) accuracy achieved, specifically 97.5% accuracy +/- 4%
  
# 6. Make Predictions
# The LDA was the most accurate model. Now we want to get an idea of the accuracy of the 
# model on our validation set.
# 
# This will give us an independent final check on the accuracy of the best model. 
# It is valuable to keep a validation set just in case you made a slip during such as 
# overfitting to the training set or a data leak. Both will result in an overly optimistic result.
# 
# We can run the LDA model directly on the validation set and summarize the results in a 
# confusion matrix.

# estimate skill of LDA on the validation dataset
predictions <- predict(fit.lda, validation)
confusionMatrix(predictions, validation$Species)

# We can see that the accuracy is 100%. It was a small validation dataset (20%), but this result 
# is within our expected margin of 97% +/-4% suggesting we may have an accurate and a reliably 
# accurate model.



  






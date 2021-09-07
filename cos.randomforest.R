library(tidyverse)
library(randomForest)
library(caret)
library(pROC)
#新数据
#newdata<-read_csv("C:/Users/gyrrx/Desktop/kaggle/output.csv")#
#修改feature engineering后#

#newdata<-out_data%>%filter(parentesco_int == 1)#
newdata<-out_data
head(newdata)
str(newdata)

#分成train和test
newtrain<-newdata[complete.cases(newdata),]
nrow(newtrain)
head(newtrain)
str(newtrain)
newtest<-newdata[!complete.cases(newdata),]
nrow(newtest)
head(newtest)
str(newtest)
train<-select(train,-Id)
train<-select(train,-idhogar)
head(newtrain)
##Create evaluation function 
F_measSummary <- function(data, lev = NULL, model = NULL) {
  cm = table(data = data$pred, reference = data$obs)
  precision <- diag(cm) / rowSums(cm)
  recall <- diag(cm) / colSums(cm)
  f1 <-  ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
  f1[is.na(f1)] <- 0
  out <- mean(f1)
  names(out) <- "F_meas"
  out
}
#target变成factor

newtrain$Target <- as.factor(newtrain$Target)
table(newtrain$Target)

# Data Partition: 70 training and 30 test
set.seed(123)

# Randomly generate label index, with 1 training and 2 test
labelindex <- sample(2, nrow(newtrain), replace = TRUE, prob = c(0.7, 0.3))


table(labelindex)
# We will sample from 1 and 2 with replacement, 
# with prob 0.7 for 1 (training data), 03 for 2 (test data)
rf_train <- newtrain[labelindex==1,]
rf_test <- newtrain[labelindex==2,]
str(rf_train)
plot(rf_train$Target,col='red')
#subsampling
set.seed(9560)
up_train <- upSample(x = rf_train[, -ncol(rf_train)],
                     y = rf_train$Target)                         
table(up_train$Target) 
plot(up_train$Target, col = "blue")

down_train <- downSample(x = rf_train[, -ncol(rf_train)],
                     y = rf_train$Target)                         
table(down_train$Target) 
plot(down_train$Target, col = "green")
#use Random Forest model

rf<-randomForest(Target~.,data=rf_train)
print(rf)

#最important的变量
varImpPlot(rf,
           sort = T,
           main =  "Variable Importance")

#feature selection
# Remove low variance features

near_zero_variance = nearZeroVar(
  x = rf_train, 
  freqCut = 99/1, 
  uniqueCut = 10, 
  saveMetrics = FALSE)

out_light = rf_train[,-near_zero_variance]

str(out_light)
# Remove highly correlated variables

#out_cor = cor(out_light)
#out_cor[is.na(out_cor)] <- 0

#near_perfect_correlation = findCorrelation(
  #x = out_cor,
  #cutoff = 0.99,
  #verbose = FALSE

#)
#out_light = out_light[,-near_perfect_correlation]
# Grab the right columns

cols_to_keep = names(out_light)
rf_train = rf_train %>% select(one_of(cols_to_keep))

xx  = c("meaneduc","dependency","overcrowding","SQBedjefe","agesq","hogar_nin")
trainset = rf_train[,!names(rf_train) %in% xx]
str(trainset)

#tuning and upsampling For Class Imbalances#
fitControl <- trainControl(method = "cv", 
                           number = 5,
                           sampling = "up",
                           summaryFunction = F_measSummary)

orifitControl <- trainControl(method = "cv", 
                              number = 5,
                              summaryFunction = F_measSummary)
set.seed(123)
#grid <- expand.grid(mtry=c(8,16,32,65))#
#after tuning,we choose mtry=32#

grid <- expand.grid(mtry=c(32))
ori_rf<-train(Target ~ ., data=trainset, 
              method = "rf", 
              metric = "F_meas",
              trControl = orifitControl,
              tuneGrid=grid)
ori_rf
randomforest_train<- train(Target ~ ., data=trainset, 
                           method = "rf", 
                           metric = "F_meas",
                           trControl = fitControl,
                           tuneGrid=grid)
randomforest_train


#Prediction in test
ori_test<-predict(ori_rf,rf_test)
prediction_test <- predict(randomforest_train,rf_test)

head(prediction_test)

#ground truth
head(rf_test$Target)
confusionMatrix(ori_test, rf_test$Target)
confusionMatrix(prediction_test, rf_test$Target)
#ROC曲线
#install.packages('pROC')
prediction_test<-as.numeric(prediction_test)
head(prediction_test)
modelroc<- roc(rf_test$Target,prediction_test)

plot(modelroc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"),smooth=TRUE, max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)
#ROC曲线
#install.packages('pROC')
prediction_test<-as.numeric(prediction_test)
ori_test<-as.numeric(ori_test)
head(prediction_test)
roc1<- roc(rf_test$Target,prediction_test)
roc2<-roc(rf_test$Target,ori_test)
plot(roc1, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE,smooth=TRUE)
plot(roc2, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE,smooth=TRUE)
plot.roc(roc2,add=T,col="yellow",print.auc=TRUE,print.auc.x=0.3,print.auc.y=0.3)

#final Prediction
randomforest_final<- train(Target ~ ., data=newtrain, 
                           method = "rf", 
                           metric = "F_meas",
                           trControl = fitControl,
                           tuneGrid=grid)
prediction_final <- predict(randomforest_final,newtest)
newtest$Target <-prediction_final

#final Prediction
randomforest_final<- train(Target ~ ., data=newtrain, 
                           method = "rf", 
                           metric = "F_meas",
                           trControl = fitControl,
                           tuneGrid=grid)
prediction_final <- predict(randomforest_final,newtest)
newtest$Target <-prediction_final
#Prepare submission file

submission <- newtest %>% select(Id,Target)

head(submission)
#Write results 
write_csv(submission,"submission.csv")


##take too much times!fit control of caret
#fitControl <- trainControl(method = "repeatedcv", 
                          # number = 10,repeats=10,
                          # summaryFunction = F_measSummary)




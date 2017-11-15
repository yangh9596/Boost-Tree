# ---------------------------------
# Data set: project evaluation
# CART decision tree and Boost tree
# Created by Yang Hu
# on Feb 25, 2017
# ---------------------------------

# Preprocessing
# ---------------------------------
setwd("/Users/yangh/Desktop/CICC")
dat <- read.csv("data.csv", header = TRUE, sep = ",") # read the data set
xnam <- paste0("x",1:29)
colnames(dat) <- c("X","Y","Z",xnam)

# Standardization 
standardize <- function(avector){
  mu <- mean(avector)
  maximum <- max(avector)
  minimum <- min(avector)
  sapply(avector, FUN = function(x){(x-mu)/(maximum-minimum)})
}
# Columns that need standardization
id <- c(1,3,4,5,7,8,9,10,11,12,13,14,15,16,17,25,26,27,28,29,30,31,32)
for(i in id){
  dat[,i] <- standardize(dat[,i])
}

# Divide test set and train set
set.seed(0)
train.index <- sample(nrow(dat), round(7*nrow(dat)/10))
train <- dat[train.index,]
test <- dat[-train.index,]

library("rpart")
library("rpart.plot")
# ---------------------------------
# Decision Tree (Y)
set.seed(0)
ct <- rpart.control(xval=10, cp=0.001, maxdepth = 30)  
xnam <- paste0("x",1:29)
fmla <- as.formula(paste("Y~",paste(xnam, collapse = "+")))
fitY <- rpart(fmla,  
             data = train[,c(-1,-3)], method = "class", control=ct,  
             parms = list(split = "gini")) 

rpart.plot(fitY, branch=1, branch.type=1, type=1, extra=103,  
           shadow.col="grey", box.col="yellow",  
           border.col="black", split.col="red",  
           split.cex=1.2)
# show results
printcp(fitY)
plotcp(fitY)

# prune tree
pfitY <- prune(fitY, cp=fitY$cptable[which.min(fitY$cptable[,"xerror"]),"CP"])
rpart.plot(pfitY, branch=1, branch.type=2, type=1, extra=102,  
           shadow.col="gray", box.col="yellow",  
           border.col="black", split.col="red",  
           split.cex=1.2)
printcp(pfitY)

# Model evaluation
predtest <- predict(pfitY,test[,c(-1,-3)])
Yresult <- apply(predtest,MARGIN = 1, FUN=function(x){which.max(x)})-1
Yerrortable <- table(test$Y,Yresult)

# error rate of prediction of Y
errorY <- 1- Yerrortable[2]/nrow(dat)

# precision rate of Y
PY <- Yerrortable[1,1]/(Yerrortable[1,1]+Yerrortable[2,1])
# recall rate of Y
RY <- Yerrortable[1,1]/(Yerrortable[1,1]+Yerrortable[1,2])
# accuracy indication F1
F1 <- 2*PY*RY/(PY+RY)

# --------------------------------
# Boosting tree
library("gbm")

# X
fmla <- as.formula(paste("X~",paste(xnam, collapse = "+")))
set.seed(0) # make the model reproducible
gbmX <- gbm(fmla,distribution = "gaussian",
            data = train[,c(-2,-3)], n.trees = 1000,
            shrinkage = 0.005, cv.folds = 5,
            interaction.depth = 2, bag.fraction = 0.5,
            verbose = FALSE)

best.iter <- gbm.perf(gbmX, method = "cv")
print(best.iter)

SE <- sum((gbmX$fit-train$X)^2)/nrow(train)# square error
summary(gbmX,n.trees = best.iter)
print(SE)
f.predict <- predict(gbmX, test[,c(-2,-3)], best.iter)
SE <- sum((f.predict-test$X)^2)/nrow(test)
print(SE)

# Z
set.seed(0)
fmla <- as.formula(paste("Z~",paste(xnam, collapse = "+")))
gbmZ <- gbm(fmla,distribution = "gaussian",
            data = train[,c(-1,-2)], n.trees = 1000,
            shrinkage = 0.005, cv.folds = 5,
            interaction.depth = 2, bag.fraction = 0.5,
            verbose = FALSE)
best.iter <- gbm.perf(gbmZ, method = "cv")
print(best.iter)


# best iter = 377
SE <- sum((gbmZ$fit-train$Z)^2)/nrow(train)# train square error
summary(gbmZ,n.trees = best.iter)
print(SE)
f.predict <- predict(gbmZ, test[,c(-2,-3)], best.iter)
SE <- sum((f.predict-train$Z)^2)/nrow(test)# test square error
print(SE)


# Overall evaluation

# --------------------------------
alpha <- 1
beta <- 0.0
gamma <- 0
score <- alpha*dat[,1]+ beta*(1-dat[,2]) + gamma*dat[,3]

# X
fmla <- as.formula(paste("X~",paste(xnam, collapse = "+")))
set.seed(0) # make the model reproducible
gbmX <- gbm(fmla,distribution = "gaussian",
            data = dat[,c(-2,-3)], n.trees = 1000,
            shrinkage = 0.005, cv.folds = 5,
            interaction.depth = 2, bag.fraction = 0.5,
            verbose = FALSE)

best.iter <- gbm.perf(gbmX, method = "cv")

# Z
set.seed(0)
fmla <- as.formula(paste("Z~",paste(xnam, collapse = "+")))
gbmZ <- gbm(fmla,distribution = "gaussian",
            data = dat[,c(-1,-2)], n.trees = 1000,
            shrinkage = 0.005, cv.folds = 5,
            interaction.depth = 2, bag.fraction = 0.5,
            verbose = FALSE)



# ---------------------------------
# Y
set.seed(0)
ct <- rpart.control(xval=10, cp=0.001, maxdepth = 30)  
xnam <- paste0("x",1:29)
fmla <- as.formula(paste("Y~",paste(xnam, collapse = "+")))
fitY <- rpart(fmla,  
              data = dat[,c(-1,-3)], method = "class", control=ct,  
              parms = list(split = "gini")) 

pfitY <- prune(fitY, cp=fitY$cptable[which.min(fitY$cptable[,"xerror"]),"CP"])
predtest <- predict(pfitY,dat[,c(-1,-3)])
Yresult <- apply(predtest,MARGIN = 1, FUN=function(x){which.max(x)})-1

f.score <- alpha*gbmX$fit+ beta*(1-Yresult) + gamma*gbmZ$fit



# ---------------------------------
# quantiles
q <- c(0,0.1,0.2,0.3,0.4,0.6,0.8,1.0)

qX <- quantile(gbmX$fit,q)
qX[1]-0.001
qY <- quantile(Yresult,q)
qY[1]-0.001
qZ <- quantile(gbmZ$fit,q)
qZ[1]-0.001
qScore <- quantile(f.score,q)
qScore[1] -0.001
meanX <- vector(mode = "numeric", length = 7)
meanY <- vector(mode = "numeric", length = 7)
meanZ <- vector(mode = "numeric", length = 7)
meanScore <- vector(mode = "numeric", length = 7)

for(i in 2:8){
  meanX[i-1] <- mean(dat[which(dat[,1]<=qX[i]&dat[,1]>qX[i-1]),1])
  meanY[i-1] <- mean(dat[which(dat[,2]<=qY[i]&dat[,2]>=qY[i-1]),2])
  meanZ[i-1] <- mean(dat[which(dat[,3]<=qZ[i]&dat[,3]>qZ[i-1]),3])
  meanScore[i-1] <- mean(score[which(score<=qScore[i]&score>qScore[i-1])])
  
}

 plot(score,f.score)
 lines(x=score,y=score,col="red")
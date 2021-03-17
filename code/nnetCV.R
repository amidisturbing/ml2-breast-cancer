#change working directory to current dir
wd = getwd()
setwd(wd)

X <- read.csv('../data/train80.csv')[c(-1,-33)] # exclude ID and X
grp <- as.factor(X$diagnosis)

X=scale(X[,2:31])
#k=length(unique(grp))
#dat=data.frame(grp,X)

n=nrow(X)
ntrain=round(n*2/3)
#install.packages('chemometrics')
require(chemometrics)
require(nnet)
set.seed(123)
train=sample(1:n,ntrain)
resnnet=nnetEval(X,grp,train,decay=c(0,0.01,0.1,0.15,0.2,0.3,0.5,1),
                 size=20,maxit=20, plotit = TRUE, legend = TRUE, legpos = "bottomright")
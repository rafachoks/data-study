
#installing packages

#install.packages("ggplot2")
#install.packages("caTools")
#install.packages("data.table")

#Introdução
#Let's explore a Kraggle database
#The goal is to make a prediction of income class (above 50k) based on each individual's information

#loading dataset
#To avoid problems delay re-encoding of strings by using stringsAsFactors = FALSE
adult <- read.csv("C:/Users/rafae/DataspellProjects/TitanicData/Adult/adult.csv/adult.csv", stringsAsFactors = TRUE)
str(adult)

#analysing the dataset
summary(adult)

#removing fnlwgt, we don't need for this analysis.
adult_filter <- subset(adult, select=c(income,age,education,educational.num,race,gender, relationship,marital.status,hours.per.week, occupation, workclass, capital.gain, capital.loss, native.country))
attach(adult_filter)

# The pre-processing

#checking for null parameters
sapply(adult_filter, function (x)all(is.na(x)))

#checking the target variable
table(income)
prop.table(table(income))
# almost 76% of the dataset earn less than 50k and 24% earn more than 50k

#analysing the data with the parameter Age
par(mfrow=c(1,2))
hist(age, main = "distribution per age")
boxplot(age~income, range = 3, main = "Checking with age parameter", col = c('gray','darkgreen'), cex=0.5)

#analysing the data using the education info.
plot(as.factor(adult_filter$educational.num), adult_filter$income,
     ylab = 'income',
     xlab = 'educational',
     col = c('darkgreen', 'gray'), cex = 0.4)


adult_filter$education_treated <- cut(adult$educational.num,
                                      breaks = c(0,8,9,10,12,16),
                                      labels = c("Ate 12th", "HS-grad", "Some-College", "Assoc", "Especialista"))

plot(as.factor(adult_filter$education_treated), adult_filter$income,
     ylab = "income",
     xlab = "Proporção de classe  de renda por educação", col = c("purple", "white"))

adult_filter$especialista <- ifelse(adult_filter$educational.num >= 13,1,0)

plot(as.factor(adult_filter$especialista), adult_filter$income,
     ylab = "Income",
     xlab = "Salario Especialista",
     col = c('darkgreen','white'))

plot(as.factor(adult_filter$gender), adult_filter$income,
     ylab = "Income", xlab = "Por Genero", col = c('red','blue'))

adult_filter$gender_male <- ifelse(adult_filter$gender == "Male", 1, 0)

plot(as.factor(adult_filter$gender_male), adult_filter$income, ylab = "Income",
     xlab = "by gender", col = c("darkgreen", "white"), cex = 0.4)

plot(as.factor(adult_filter$relationship), adult_filter$income,
     ylab = "income",
     xlab = "Proporção por relacionamento",
     col = c("darkgreen", "gray"),
     cex = 0.4)

adult_filter$married <- ifelse(adult_filter$relationship == "Husband" | 
                                 adult_filter$relationship == "Wife",1,0)


plot(as.factor(adult_filter$married), adult_filter$income, ylab = "Income",
     xlab = "Casado", col = c("darkgreen","gray"), cex = 0.4)

plot(as.factor(adult_filter$occupation), adult_filter$income,
     ylab = "Income", xlab = "by Occupation", col = c("darkgreen", "gray"), cex = 0.1 )

summary(adult_filter$occupation)

adult_filter$occupation_treated <- ifelse(as.character(adult_filter$occupation) == "Exec-managerial" |
                                            as.character(adult_filter$occupation) == "Prof-specialty" |
                                            as.character(adult_filter$occupation) == "Armed-Forces" |
                                            as.character(adult_filter$occupation) == "Tech-support" |
                                            as.character(adult_filter$occupation) == "Protective-serv" |
                                            as.character(adult_filter$occupation) == "Sales", 1, 0)

plot(as.factor(adult_filter$occupation_treated), adult_filter$income, ylab = "Income",
     xlab = "by occupation", col = c("darkgreen", "gray"), cex = 0.4)

attach(adult_filter)

par(mfrow = c(1,2))

hist(hours.per.week, cex=0.4)
boxplot(hours.per.week~income, range = 3, main = "hours per week", col = c("darkgreen", "gray"), cex = 0.4)

adult_filter$work_hours <- cut(adult$hours.per.week, breaks = c(0,39,40,100), labels = c("Until 40H", "40H", "More than 40H"))

plot(as.factor(adult_filter$work_hours), adult_filter$income, ylab = "Income", xlab = "Work Hours", cex = 0.4)

adult_filter$hightWH <- ifelse(adult_filter$hours.per.week > 40, 1, 0)

adult_filter$cat_cap.gain <- ifelse(as.numeric(capital.gain) >= 1 , 1,0)
adult_filter$cat_cap.loss <- ifelse(as.numeric(capital.loss) >= 1 , 1,0)

str(adult_filter)

Base_Fim <- subset(adult_filter, select = c(income, age, gender_male, especialista, married, occupation_treated, hightWH, capital.loss ))
str(Base_Fim)
summary(Base_Fim)


Base_Fim[, 2:2] = scale(Base_Fim[, 2:2])
summary(Base_Fim)
attach(Base_Fim)


#transformando modelo



library(caTools)

set.seed(21)
split <- sample.split(Base_Fim$income, SplitRatio = 0.7)

train_adult <- subset(Base_Fim, split == TRUE)
test_adult <- subset(Base_Fim, split == FALSE)


#rodando KNN
library(class)

prvisao_KNN <- knn(train_adult[,-1], test_adult[,-1], cl = train_adult$income, k = 41)

n_base_previsao <- data.frame(prvisao_KNN)

library(caret)

MC_Knn <- table(prvisao_KNN, test_adult$income, deparse.level = 2)

confusionMatrix(MC_Knn, positive = ">50K")


#utilizando cross validation

TrainClasse <- train_adult[,1]

#encontrando o melhro K vizionhos para o modelo

Knnfit <- train(train_adult[,2:8], TrainClasse,
                method = "knn",
                tuneLength = 20,
                trControl = trainControl(method = "cv"))



Knnfit










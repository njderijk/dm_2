library(rpart.plot)
library(tm)
library(SnowballC)
library(randomForest)
library(tidyverse)
library(quanteda)
library(tidytext)
library(qdap)
library(ggplot2)
library(read_text)
library(textdata)
library(janeaustenr)
library(dplyr)
library(stringr)
library(readtext)
library(syuzhet)
library(ngram)
library(caret)
library(tree)
library(glmnet)

# Load the original data
list_of_files_d <- list.files(path = "./deceptive/allfortm_d", recursive = TRUE,
                              pattern = "*.txt", 
                              full.names = TRUE)
txt_d <- readtext(paste0(list_of_files_d))

list_of_files_t <- list.files(path = "./truthful/allfortm_t", recursive = TRUE,
                              pattern = "*.txt", 
                              full.names = TRUE)
txt_t <- readtext(paste0(list_of_files_t))

list_of_files_test <- list.files(path = "./test", recursive = TRUE,
                                 pattern = "*.txt", 
                                 full.names = TRUE)
txt_test <- readtext(paste0(list_of_files_test))
labels_test <- c(rep("deceptive", 80), rep("truthful",80))

# complete text
txt_c <- bind_rows(txt_t, txt_d)
labels <- c(rep("truthful",320), rep("deceptive",320))

# Complete text with labels [640x3]
txt_c <- txt_c %>%
  add_column(labels = factor(labels)) 

# Complete test set with labels 
txt_test <- txt_test %>%
  add_column(labels = factor(labels_test))

### CREATE TRAIN / TEST 
### 640 (truthful, deceptive) - 180 test (deceptive, truthful)
txt_full <- rbind(txt_c,txt_test)

n_gram_list_full_2b <- txt_full %>%
  select(c("text", "labels", "doc_id")) %>%
  unnest_tokens(output = word, input = text, token = "ngrams", n =2) %>%
  separate(word, c("word1", "word2"), sep = " ") %>%
  filter(
    !word1 %in% stop_words$word,                 # remove stopwords from both words in bi-gram
    !word2 %in% stop_words$word,
    !str_detect(word1, pattern = "[[:digit:]]"), # removes any words with numeric digits
    !str_detect(word2, pattern = "[[:digit:]]"),
    !str_detect(word1, pattern = "[[:punct:]]"), # removes any remaining punctuations
    !str_detect(word2, pattern = "[[:punct:]]"),
    !str_detect(word1, pattern = "(.)\\1{2,}"), # removes any words with 3 or more repeated letters
    !str_detect(word2, pattern = "(.)\\1{2,}"),
    !str_detect(word1, pattern = "\\b(.)\\b"), # removes any remaining single letter words
    !str_detect(word1, pattern = "\\b(.)\\b")
  ) %>%
  unite("word", c(word1, word2), sep = "") 

# create ngram - 1 features
n_gram_list_full_1 <- txt_full %>%
  select(c("text", "labels", "doc_id")) %>% 
  unnest_tokens(output = word, input = text) %>%
  filter(
    !str_detect(word, pattern = "[[:digit:]]"), # removes any words with numeric digits
    !str_detect(word, pattern = "[[:punct:]]"), # removes any remaining punctuations
    !str_detect(word, pattern = "(.)\\1{2,}"), # removes any words with 3 or more repeated letters
    !str_detect(word, pattern = "\\b(.)\\b"), # removes any remaining single letter words
  ) %>%
  anti_join(get_stopwords())

# Complete N-gram list with the bigrams and ngrams
full_set <- rbind(n_gram_list_full_1, n_gram_list_full_2b)
# full_set <- n_gram_list_full_1 ## unigrams

# dtm
dtm_full <- full_set %>%
  count(doc_id, word) %>%
  cast_dtm(document = doc_id, term = word, value = n)

dtm_full_sparser <- removeSparseTerms(dtm_full, .99 )

dtm_full_single_words <- n_gram_list_full_1 %>%
  count(doc_id, word) %>%
  cast_dtm(document = doc_id, term = word, value = n)

dtm_full_test <- full_set %>%
  filter(doc_id %in% txt_test$doc_id) %>%
  count(doc_id, word) %>%
  cast_dtm(document = doc_id, term = word, value = n)

dtm_test <- dtm_full[dtm_full$dimnames$Docs %in% txt_test$doc_id,]
dtm_training <- dtm_full[!dtm_full$dimnames$Docs %in% txt_test$doc_id,]

# dataframes
df_full <- full_set %>%
  count(doc_id, word) %>% # count
  spread(word, n) %>% # convert to wide format
  map_df(replace_na, 0)

df_training <- full_set %>%
  count(doc_id, word) %>%
  pivot_wider(names_from = word, values_from = n) %>%
  map_df(replace_na, 0) %>%
  filter(!doc_id %in% txt_test$doc_id) %>%
  select(-doc_id)

df_test <- full_set %>%
  count(doc_id, word) %>%
  pivot_wider(names_from = word, values_from = n) %>%
  map_df(replace_na, 0) %>%
  filter(doc_id %in% txt_test$doc_id) %>%
  select(-doc_id)

#Use the sparse format: to remove features
df_full_2 <- tidy(dtm_full_sparser) %>%
  count(document, term) %>% # count
  spread(term, n) %>% # convert to wide format
  map_df(replace_na, 0)

df_training <- df_full_2 %>%
  filter(!document %in% txt_test$doc_id) %>%
  select(-document)

df_test <- df_full_2 %>%
  filter(document %in% txt_test$doc_id) %>%
  select(-document)

test_labels <- as.factor(txt_test$labels)
training_labels <- as.factor(c(rep("deceptive",320), rep("truthful",320)))

## USE DF TO TRAIN CLASSIFICATION TREES

# Random Forest without bigrams (No Tuning)
classifier <- randomForest(x = df_training, 
                           y = training_labels,
                           nTree = 10)

y_pred <- predict(classifier, newdata = df_test)

conf_matrix_rf <- table(true = c(rep("deceptive", 80), rep("truthful", 80)), pred = y_pred)
conf_matrix_rf

# accuracy, precision, recall, F1
TP <- conf_matrix_rf[1,1]
FP <- conf_matrix_rf[1,2]
FN <- conf_matrix_rf[2,1]
TN <- conf_matrix_rf[2,2]

accuracy <- (TP + TN) / (TP + TN + FP + FN)
precision <- (TP) / (TP + FP)  
F1 <- (2 * TP) / (2 * TP + FP + FN)
recall = TP/(TP+FN)

print(c(accuracy, precision, recall, F1))



# Random Forest without Bigrams
set.seed(34)
#create a task
traintask <- makeClassifTask(data = df_training, target = "labels")
testtask <- makeClassifTask(data = df_test, target = "labels")

# set 5-fold cross validation
rdesc <- makeResampleDesc("CV",iters=5L)

# detect cores
parallelStartSocket(cpus = detectCores())

# make randomForest learner
rf.lrn <- makeLearner("classif.randomForest")
rf.lrn$par.vals <- list(ntree = 25L, importance=TRUE)
r_rf <- resample(learner = rf.lrn, task = traintask, resampling = rdesc, measures = list(tpr,fpr,fnr,fpr,acc), show.info = T)

# Hyperparameter tuning
getParamSet(rf.lrn)

#set parameter space
params <- makeParamSet(makeIntegerParam("mtry",lower = 1,upper = 250),makeIntegerParam("nodesize", lower = 1,upper = 50), makeIntegerParam("ntree", lower = 1,upper = 25))

#set validation strategy
rdesc <- makeResampleDesc("CV",iters=5L)

#set optimization technique
ctrl <- makeTuneControlRandom(maxit = 5L)

#start tuning
tune <- tuneParams(learner = rf.lrn, task = traintask, resampling = rdesc, measures = list(acc), par.set = params, control = ctrl, show.info = T)
# [Tune] Result: mtry=6; nodesize=44; ntree=17 : acc.test.mean=0.7421875

print(c("Optimal parameters:", tune$x))


classifier_hp <- randomForest(x = df_training, 
                              y = training_labels,
                              mtry = 207,
                              nodesize = 19,
                              ntree = 8)

y_pred <- predict(classifier_hp, newdata = df_test)

conf_matrix_rf <- table(true = c(rep("deceptive", 80), rep("truthful", 80)), pred = y_pred)
conf_matrix_rf

# accuracy, precision, recall, F1
TP <- conf_matrix_rf[1,1]
FP <- conf_matrix_rf[1,2]
FN <- conf_matrix_rf[2,1]
TN <- conf_matrix_rf[2,2]

accuracy <- (TP + TN) / (TP + TN + FP + FN)
precision <- (TP) / (TP + FP)  
F1 <- (2 * TP) / (2 * TP + FP + FN)
recall = TP/(TP+FN)

print(c(accuracy, precision, recall, F1))


# Random Forest with Bigrams (No Tuning)
classifier <- randomForest(x = df_training, 
                           y = training_labels,
                           nTree = 10)

y_pred <- predict(classifier, newdata = df_test)

conf_matrix_rf <- table(true = c(rep("deceptive", 80), rep("truthful", 80)), pred = y_pred)
conf_matrix_rf

# accuracy, precision, recall, F1
TP <- conf_matrix_rf[1,1]
FP <- conf_matrix_rf[1,2]
FN <- conf_matrix_rf[2,1]
TN <- conf_matrix_rf[2,2]

accuracy <- (TP + TN) / (TP + TN + FP + FN)
precision <- (TP) / (TP + FP)  
F1 <- (2 * TP) / (2 * TP + FP + FN)
recall = TP/(TP+FN)

print(c(accuracy, precision, recall, F1))



# Random Forest with Bigrams (With Tuning)
set.seed(34)
#create a task
traintask <- makeClassifTask(data = df_training, target = "labels") 
testtask <- makeClassifTask(data = df_test, target = "labels")

# set 5-fold cross validation
rdesc <- makeResampleDesc("CV",iters=5L)

# detect cores
parallelStartSocket(cpus = detectCores())

# make randomForest learner
rf.lrn <- makeLearner("classif.randomForest")
rf.lrn$par.vals <- list(ntree = 25L, importance=TRUE)
r_rf <- resample(learner = rf.lrn, task = traintask, resampling = rdesc, measures = list(tpr,fpr,fnr,fpr,acc), show.info = T)

# Hyperparameter tuning
getParamSet(rf.lrn)

#set parameter space
params <- makeParamSet(makeIntegerParam("mtry",lower = 1,upper = 1000),makeIntegerParam("nodesize", lower = 1,upper = 100), makeIntegerParam("ntree", lower = 1,upper = 50))

#set validation strategy
rdesc <- makeResampleDesc("CV",iters=10L)

#set optimization technique
ctrl <- makeTuneControlRandom(maxit = 10L)

#start tuning
tune <- tuneParams(learner = rf.lrn, task = traintask, resampling = rdesc, measures = list(acc), par.set = params, control = ctrl, show.info = T)
# [Tune] Result: mtry=6; nodesize=44; ntree=17 : acc.test.mean=0.7421875

print(c("Optimal parameters:", tune$x))


classifier_hp <- randomForest(x = df_training, 
                           y = training_labels,
                           mtry = 160,
                           nodesize = 79,
                           ntree = 26)

y_pred <- predict(classifier_hp, newdata = df_test)

conf_matrix_rf <- table(true = c(rep("deceptive", 80), rep("truthful", 80)), pred = y_pred)
conf_matrix_rf

# accuracy, precision, recall, F1
TP <- conf_matrix_rf[1,1]
FP <- conf_matrix_rf[1,2]
FN <- conf_matrix_rf[2,1]
TN <- conf_matrix_rf[2,2]

accuracy <- (TP + TN) / (TP + TN + FP + FN)
precision <- (TP) / (TP + FP)  
F1 <- (2 * TP) / (2 * TP + FP + FN)
recall = TP/(TP+FN)

print(c(accuracy, precision, recall, F1))




# Classification Tree with bigrams and tuning
df_training$labels = training_labels
df_test$labels = test_labels

df_training$labels = as.factor(df_training$labels)
df_test$labels = as.factor(df_test$labels)

traintask <- makeClassifTask(data = df_training, target = "labels") 
testtask <- makeClassifTask(data = df_test, target = "labels")

# set 5-fold cross validation
rdesc <- makeResampleDesc("CV",iters=5L)

#create learner
ct <- makeLearner("classif.rpart", predict.type = "response")
ct.lrn <- makeBaggingWrapper(learner = ct, bw.iters = 25, bw.replace = TRUE)
r_ct <- resample(learner = ct.lrn , task = traintask, resampling = rdesc, measures = list(tpr,fpr,fnr,fpr,acc) ,show.info = T)

getParamSet(ct.lrn)

#set parameter space
params <- makeParamSet(makeIntegerParam("minsplit",lower = 1,upper = 15),makeIntegerParam("maxdepth",lower = 1,upper = 25))

#set validation strategy
rdesc <- makeResampleDesc("CV",iters=5L)

#set optimization technique
ctrl <- makeTuneControlRandom(maxit = 5L)

#start tuning
tune_ct <- tuneParams(learner = ct.lrn, task = traintask, resampling = rdesc, measures = list(acc), par.set = params, control = ctrl, show.info = T)
# [Tune] Result: minsplit=2; maxdepth=25 : acc.test.mean=0.7531250

ct_tuned = rpart(labels ~ ., data=df_training, method="class", minsplit=3, maxdepth=15)
predictCT_tuned = predict(ct_tuned, newdata=df_test, type="class")

# Model accuracy Classification Tree (Tuned)
conf_matrix_ct <- table(true = txt_test$labels, pred = predictCT_tuned)

TP_ct <- conf_matrix_ct[1,1]
FP_ct <- conf_matrix_ct[1,2]
FN_ct <- conf_matrix_ct[2,1]
TN_ct <- conf_matrix_ct[2,2]

accuracy_ct <- (TP_ct + TN_ct) / (TP_ct + TN_ct + FP_ct + FN_ct)
precision_ct <- (TP_ct) / (TP_ct + FP_ct)  
F1_ct <- (2 * TP_ct) / (2 * TP_ct + FP_ct + FN_ct)
recall_ct = TP_ct/(TP_ct+FN_ct)

print(c(accuracy_ct, precision_ct, recall_ct, F1_ct))

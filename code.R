library(rpart.plot)
library(tm)
library(SnowballC)
library(randomForest)

txt_d <- Corpus(DirSource("~/deceptive/allfortm_d"), readerControl = list(language="lat")) #specifies the exact folder where my text file(s) is for analysis with tm.
txt_t <- Corpus(DirSource("~/truthful/allfortm_t"), readerControl = list(language="lat"))

corpus <- Corpus(DirSource("~/all"), readerControl = list(language="lat"))

# Process
corpus <- tm_map(txt_all, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus , stripWhitespace)
corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stemDocument)

dtm <- DocumentTermMatrix(corpus)
sparse <- removeSparseTerms(corpus, 0.995)
freq <- DocumentTermMatrix(corpus)



tSparse = as.data.frame(as.matrix(sparse))
colnames(tSparse) = make.names(colnames(sparse))
for (i in row.names(tSparse)) {
  if (startsWith(i, "t_")) {
    tSparse[i,]$class = 1
  }
  else {
    tSparse[i,]$class = 0
  }
}

# Determine baseline accuracy (0.50 / 50%)
prop.table(table(tSparse$class))

trainSparse = tSparse[!complete.cases(tSparse)] 
droprecords <-  tSparse[complete.cases(tSparse)] 

RF_model = randomForest(class ~ ., data=trainSparse)
# predictRF = predict(RF_model, newdata=testSparse)


# # Process deceptive reviews
# txt_d <- tm_map(txt_d, removeNumbers)
# txt_d <- tm_map(txt_d, removePunctuation)
# txt_d <- tm_map(txt_d , stripWhitespace)
# txt_d <- tm_map(txt_d, tolower)
# txt_d <- tm_map(txt_d, removeWords, stopwords("english"))
# txt_d <- tm_map(txt_d, stemDocument)
# 
# txt_d_dtm <- DocumentTermMatrix(txt_d)
# sparse_d <- removeSparseTerms(txt_d_dtm, 0.90)
# 
# freq_d <- DocumentTermMatrix(txt_d)
# 
# 
# tSparse_d = as.data.frame(as.matrix(sparse_d))
# colnames(tSparse_d) = make.names(colnames(tSparse_d))
# tSparse_d$class = 1
# 
# # Process truthful reviews
# txt_t <- tm_map(txt_t, removeNumbers)
# txt_t <- tm_map(txt_t, removePunctuation)
# txt_t <- tm_map(txt_t , stripWhitespace)
# txt_t <- tm_map(txt_t, tolower)
# txt_t <- tm_map(txt_t, removeWords, stopwords("english"))
# txt_t <- tm_map(txt_t, stemDocument)
# 
# txt_t_dtm <- DocumentTermMatrix(txt_t)
# sparse_t <- removeSparseTerms(txt_t_dtm, 0.90)
# 
# freq_t <- DocumentTermMatrix(txt_t)
# 
# tSparse_t = as.data.frame(as.matrix(sparse_t))
# colnames(tSparse_t) = make.names(colnames(tSparse_t))
# tSparse_t$class = 1
# 
# # Append data frames
# 
# tSparse_all <- Append(tSparse_d, tSparse_t)

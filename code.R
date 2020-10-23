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

train.mnb <- function (dtm,labels) 
{
  call <- match.call()
  V <- ncol(dtm)
  N <- nrow(dtm)
  prior <- table(labels)/N
  labelnames <- names(prior)
  nclass <- length(prior)
  cond.probs <- matrix(nrow=V,ncol=nclass)
  dimnames(cond.probs)[[1]] <- dimnames(dtm)[[2]]
  dimnames(cond.probs)[[2]] <- labelnames
  index <- list(length=nclass)
  for(j in 1:nclass){
    index[[j]] <- c(1:N)[labels == labelnames[j]]
  }
  
  for(i in 1:V){
    for(j in 1:nclass){
      cond.probs[i,j] <- (sum(dtm[index[[j]],i])+1)/(sum(dtm[index[[j]],])+V)
    }
  }
  list(call=call,prior=prior,cond.probs=cond.probs)    
}

predict.mnb <-
  function (model,dtm) 
  {
    classlabels <- dimnames(model$cond.probs)[[2]]
    logprobs <- dtm %*% log(model$cond.probs)
    N <- nrow(dtm)
    nclass <- ncol(model$cond.probs)
    logprobs <- logprobs+matrix(nrow=N,ncol=nclass,log(model$prior),byrow=T)
    classlabels[max.col(logprobs)]
  }

# Create corpus
# Negative deceptive 
corpus_d <- Corpus(DirSource("./deceptive/allfortm_d"), readerControl = list(language="lat"))
# Negative truthful 
corpus_t <- Corpus(DirSource("./truthful/allfortm_t"), readerControl = list(language="lat"))

#Negative truthful and deceptive both. 
corpus <- Corpus(DirSource("./all"), readerControl = list(language="lat"))

# Load the original data
list_of_files_d <- list.files(path = "./deceptive/allfortm_d", recursive = TRUE,
                              pattern = "*.txt", 
                              full.names = TRUE)
txt_d <- readtext(paste0(list_of_files_d))

list_of_files_t <- list.files(path = "./truthful/allfortm_t", recursive = TRUE,
                              pattern = "*.txt", 
                              full.names = TRUE)
txt_t <- readtext(paste0(list_of_files_t))

# complete text
txt_c <- bind_rows(txt_t, txt_d)
labels <- c(rep("truthful",320), rep("deceptive",320))

# Complete text with labels [640x3]
txt_c <- txt_c %>%
  add_column(labels = factor(labels)) 

stopwords <- tm::stopwords(kind = "en")

# Prepare for word_cloud (no further preprocessing: needs to remove stopwords)
dfm_corpus_t <- txt_c[txt_c$labels == "truthful",] %>%
  corpus(text_field = "text") %>%
  tokens(remove_numbers = FALSE, remove_punct = TRUE, remove_symbols = TRUE, remove_separators = TRUE, split_hyphens	
         = TRUE, remove_url = TRUE) %>% 
  dfm()

dfm_corpus_d <- txt_c[txt_c$labels == "deceptive",] %>%
  corpus(text_field = "text") %>%
  tokens(remove_numbers = FALSE, remove_punct = TRUE, remove_symbols = TRUE, remove_separators = TRUE, split_hyphens	
 = TRUE, remove_url = TRUE) %>%
  dfm()

# Textplot wordcloud
set.seed(100)
textplot_wordcloud(dfm_corpus_t, min_count = 100, random_order = FALSE,
                   rotation = .25, 
                   color = RColorBrewer::brewer.pal(8,"Dark2"))

set.seed(100)
textplot_wordcloud(dfm_corpus_d, min_count = 250, random_order = FALSE,
                   rotation = .25, 
                   color = RColorBrewer::brewer.pal(8,"Dark2"))

### Tokenize the data (ALL THE DATA)
cleaned_data_tokens <- txt_c %>% 
  select(c("text", "labels", "doc_id")) %>% 
  unnest_tokens(output = word, input = text)

cleaned_data_tokens_nosw <- cleaned_data_tokens %>%
  anti_join(get_stopwords())

# Tf_idf 
cleaned_data_tokens_nosw_t <- cleaned_data_tokens_nosw[labels == "truthful",]
cleaned_data_tokens_nosw_d <- cleaned_data_tokens_nosw[labels == "deceptive",]

tf_idf_t <- cleaned_data_tokens_nosw_t %>%
  count(doc_id, word, sort = TRUE) %>%
  bind_tf_idf(word, doc_id, n)

tf_idf_d <- cleaned_data_tokens_nosw_d %>%
  count(doc_id, word, sort = TRUE) %>%
  bind_tf_idf(word, doc_id, n)

tf_idf_t %>%
  top_n(50, tf_idf) %>%
  ggplot(aes(x = tf_idf, y = word)) + geom_col()

tf_idf_d %>%
  top_n(50, tf_idf) %>%
  ggplot(aes(x = tf_idf, y = word)) + geom_col()

# Create dtm
class_dtm_c <- cleaned_data_tokens_nosw %>%
  cast_dtm(document = labels, term = word, value = n)

class_dtm_c

# Reducing model complexity by removing sparse terms from the model: tokens that do not appear across many documents 
class_dtm_c2 <- removeSparseTerms(class_dtm_c, sparse =.99) #97 terms 562
class_dtm_c2
# removeSparseTerms(class_dtm, sparse =.98) #96 terms 259
# removeSparseTerms(class_dtm, sparse =.97) #97 terms 158

# Plot the occurences of the word for each label
type_words <-  cleaned_data_tokens_nosw %>%
  count(labels, word, sort = TRUE)

total_words <- type_words %>% 
  group_by(labels) %>% 
  summarize(total = sum(n))

type_words <- left_join(type_words, total_words)
type_words

# Term frequency
freq_by_rank <- type_words %>% 
  group_by(labels) %>% 
  mutate(rank = row_number(), 
         tf = n/total)

# plot the counts
type_words %>%
  group_by(labels) %>%
  top_n(n=50,n) %>% 
  ggplot(aes(x= word, y = n)) +
  geom_col() +
  labs(x = NULL, y = "n") +
  facet_wrap(~ labels, scales = "free") +
  coord_flip()

# graph the top 50 tokens for 2 categories: plot the tf
freq_by_rank %>%
  top_n(50,tf) %>%
  ungroup() %>%
  ggplot(aes(word, tf)) +
  geom_col() +
  labs(x = NULL, y = "tf") +
  facet_wrap(~ labels, scales = "free") +
  coord_flip()


# Wordcloud: All docs
word_matrix <- as.matrix(TermDocumentMatrix(corpus))
words <- sort(rowSums(word_matrix),decreasing=TRUE) 
word_df <- data.frame(word = names(words),freq=words)
wordcloud(words = word_df$word, freq = word_df$freq, min.freq = 100, max.words=200, random.order=FALSE, rot.per=0.35, colors=brewer.pal(8, "Dark2"))

# Wordcloud: Deceptive docs
corpus_d <- tm_map(corpus_d, removeNumbers)
corpus_d <- tm_map(corpus_d, removePunctuation)
corpus_d <- tm_map(corpus_d , stripWhitespace)
corpus_d <- tm_map(corpus_d, tolower)
corpus_d <- tm_map(corpus_d, removeWords, stopwords("english"))
corpus_d <- tm_map(corpus_d, stemDocument)
dtm_d <- TermDocumentMatrix(corpus_d)
word_matrix_d <- as.matrix(dtm_d)
words_d <- sort(rowSums(word_matrix_d),decreasing=TRUE) 
word_df_d <- data.frame(word = names(words_d),freq=words_d)
wordcloud(words = word_df_d$word, freq = word_df_d$freq, min.freq = 1, max.words=200, random.order=FALSE, rot.per=0.35, colors=brewer.pal(8, "Dark2"))


# Wordcloud: Truthful docs
corpus_t <- tm_map(corpus_t, removeNumbers)
corpus_t <- tm_map(corpus_t, removePunctuation)
corpus_t <- tm_map(corpus_t , stripWhitespace)
corpus_t <- tm_map(corpus_t, tolower)
corpus_t <- tm_map(corpus_t, removeWords, stopwords("english"))
corpus_t <- tm_map(corpus_t, stemDocument)
dtm_t <- TermDocumentMatrix(corpus_t)
word_matrix_t <- as.matrix(dtm_t)
words_t <- sort(rowSums(word_matrix_t),decreasing=TRUE) 
word_df_t <- data.frame(word = names(words_t),freq=words_t)
wordcloud(words = word_df_t$word, freq = word_df_t$freq, min.freq = 1, max.words=200, random.order=FALSE, rot.per=0.35, colors=brewer.pal(8, "Dark2"))


# Process
corpus <- tm_map(txt_all, removeNumbers)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus , stripWhitespace)
corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stemDocument)

dtm <- DocumentTermMatrix(corpus)
sparse <- removeSparseTerms(dtm, 0.995)
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


# # Process all reviews

docs <- Corpus(DirSource("./all"), readerControl = list(language="lat"))
docs <- tm_map(docs, removeNumbers)
docs <- tm_map(docs, removePunctuation)
docs <- tm_map(docs , stripWhitespace)
docs <- tm_map(docs, tolower)
docs <- tm_map(docs, removeWords, stopwords("english"))
docs <- tm_map(docs, stemDocument)

dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)
v <- sort(rowSums(m), decreasing=TRUE)
d <- data.frame(words = names(v), freq=v)

# Clean
clean_text <- corpus
clean_text <- tm_map(clean_text, removeNumbers)
clean_text <- tm_map(clean_text, removePunctuation)
clean_text <- tm_map(clean_text , stripWhitespace)
clean_text <- tm_map(clean_text, tolower)
clean_text <- tm_map(clean_text, removeWords, stopwords("english"))
clean_text <- tm_map(clean_text, stemDocument)

clean_df <- data.frame(text=sapply(clean_text, identity), stringsAsFactors=F)

labels <- c(rep("truthful",320), rep("deceptive",320))

# Complete text with labels [640x3]
clean_df <- clean_df %>%
  add_column(labels = factor(labels)) 


# Sentiment Analysis for all reviews
sent_d <- get_nrc_sentiment(clean_text$text)
sent_td <- data.frame(t(sent_d))
sent_td_new <- data.frame(rowSums(sent_td[2:640]))

names(sent_td_new)[1] <- "count"
sent_td_new <- cbind("sentiment" = rownames(sent_td_new), sent_td_new)
rownames(sent_td_new) <- NULL
sent_td_new2<-sent_td_new[1:10,]

qplot(df_trigrams$ngrams, data=df_trigrams, weight=df_trigrams$prop, geom="bar")+ggtitle("Overall review sentiments")

# Sentiment Analysis for deceptive reviews
clean_deceptive <- clean_text[321:640,]
sent_d <- get_nrc_sentiment(clean_deceptive$text)
sent_dd <- data.frame(t(sent_d))
sent_dd_new <- data.frame(rowSums(sent_dd[2:320]))

names(sent_dd_new)[1] <- "count"
sent_dd_new <- cbind("sentiment" = rownames(sent_dd_new), sent_dd_new)
rownames(sent_dd_new) <- NULL
sent_dd_new2<-sent_dd_new[1:10,]

quickplot(sentiment, data=sent_dd_new2, weight=count, geom="bar", fill=sentiment, ylab="count")+ggtitle("Sentiments of deceptive reviews")


# Sentiment Analysis for truthful reviews
clean_truthful <- clean_text[1:320,]
sent_t <- get_nrc_sentiment(clean_truthful$text)
sent_td <- data.frame(t(sent_t))
sent_td_new <- data.frame(rowSums(sent_td[2:320]))

names(sent_td_new)[1] <- "count"
sent_td_new <- cbind("sentiment" = rownames(sent_td_new), sent_td_new)
rownames(sent_td_new) <- NULL
sent_td_new2<-sent_td_new[1:10,]

quickplot(sentiment, data=sent_td_new2, weight=count, geom="bar", fill=sentiment, ylab="count")+ggtitle("Sentiments of truthful reviews")


# N-grams

  # Bigrams for truthful
ng_bi_t <- ngram(clean_df[clean_df$labels == "truthful",]$text, 2)
ng_bi_t
bigrams_t <- get.phrasetable(ng_bi_t)
df_bigrams_t <- as.data.frame(bigrams_t)

par(mar=c(9,3,3,3))
barplot(df_bigrams_t$freq[0:10], names.arg=df_bigrams_t$ngrams[0:10], las=2)

  # Bigrams for deceptive
ng_bi_d <- ngram(clean_df[clean_df$labels == "deceptive",]$text, 2)
ng_bi_d
bigrams_d <- get.phrasetable(ng_bi_d)
df_bigrams_d <- as.data.frame(bigrams_d)

par(mar=c(12,4,4,4))
barplot(df_bigrams_d$freq[0:10], names.arg=df_bigrams_d$ngrams[0:10], las=2)


  # Trigram for truthful
ng_tri_t <- ngram(clean_df[clean_df$labels == "truthful",]$text, 3)
ng_tri_t
trigrams_t <- get.phrasetable(ng_tri_t)
df_trigrams_t <- as.data.frame(trigrams_t)

par(mar=c(12,4,4,4))
barplot(df_trigrams_t$freq[0:10], names.arg=df_trigrams_t$ngrams[0:10], las=2)

# Trigram for deceptive
ng_tri_d <- ngram(clean_df[clean_df$labels == "deceptive",]$text, 3)
ng_tri_d
trigrams_d <- get.phrasetable(ng_tri_d)
df_trigrams_d <- as.data.frame(trigrams_d)

par(mar=c(12,4,4,4))
barplot(df_trigrams_d$freq[0:10], names.arg=df_trigrams_d$ngrams[0:10], las=2)

# word counts per document
word_counts_t <- txt_t
for (i in 1:nrow(txt_t)){
  word_counts_t[i, 'wordcount'] <- wordcount(txt_t$text[i], sep = " ", count.function = sum)
}
word_counts_t <- word_counts_t[c('doc_id', 'wordcount')]

word_counts_d <- txt_d
for (i in 1:nrow(txt_d)){
  word_counts_d[i, 'wordcount'] <- wordcount(txt_d$text[i], sep = " ", count.function = sum)
}
word_counts_d <- word_counts_d[c('doc_id', 'wordcount')]

summary(word_counts_t)
summary(word_counts_d)

ggplot(data = word_counts_t, aes(wordcount)) + geom_histogram(color='darkblue', fill='lightblue')
ggplot(data = word_counts_d, aes(wordcount)) + geom_histogram(color='darkblue', fill='lightblue')

####### Training the models ##########
# TODO: Load the test-data

# Multinominal naive bayes:
naive_bayes <- train.mnb(dtm = class_dtm_c,labels=c('deceptive','truthful')) 

prediction <- predict.mnb(model=naive_bayes, dtm = test_dtm)


# Classification Tree
train_data <- clean_df
classification_tree = tree(formula = labels ~ text, data = train_data)
summary(classification_tree)

# Random Forest
train_data <- clean_df
set.seed(1234)
random_forest <- randomForest(formula = labels ~ ., data = clean_df)
print(random_forest)


# # Previous code
# txt_d <- Corpus(DirSource("./deceptive/allfortm_d"), readerControl = list(language="lat"))
# txt_d <- tm_map(txt_d, removeNumbers)
# txt_d <- tm_map(txt_d, removePunctuation)
# txt_d <- tm_map(txt_d , stripWhitespace)
# txt_d <- tm_map(txt_d, tolower)
# txt_d <- tm_map(txt_d, removeWords, stopwords("english"))
# txt_d <- tm_map(txt_d, stemDocument)
# 
# txt_d_dtm <- DocumentTermMatrix(txt_d)
# dtm_t <- as.matrix(txt_d_dtm)
# 
# sparse_d <- removeSparseTerms(txt_d_dtm, 0.90)
# 
# 
# tSparse_d = as.data.frame(as.matrix(sparse_d))
# colnames(tSparse_d) = make.names(colnames(tSparse_d))
# tSparse_d$class = 1
# 
# # Process truthful reviews
# txt_t <- Corpus(DirSource("./truthful/allfortm_t"), readerControl = list(language="lat"))
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
# # Append data frames# 
# tSparse_all <- Append(tSparse_d, tSparse_t)
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
  top_n(20, tf_idf) %>%
  ggplot(aes(x = tf_idf, y = word)) + geom_col()

tf_idf_d %>%
  top_n(20, tf_idf) %>%
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
corpus <- tm_map(txt_c, removeNumbers)
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


### 

# create bigrams
# n_gram_list<- txt_c %>%
#   select(c("text", "labels", "doc_id")) %>%
#   unnest_tokens(output = bigram, input = text, token = "ngrams", n =2) %>%
#   separate(bigram, c("word1", "word2"), sep = " ") %>%
#   filter(
#     !word1 %in% stop_words$word, # remove stopwords from both words in bi-gram
#     !word2 %in% stop_words$word,
#     !str_detect(word1, pattern = "[[:digit:]]"), # removes any words with numeric digits
#     !str_detect(word2, pattern = "[[:digit:]]"),
#     !str_detect(word1, pattern = "[[:punct:]]"), # removes any remaining punctuations
#     !str_detect(word2, pattern = "[[:punct:]]"),
#     !str_detect(word1, pattern = "(.)\\1{2,}"), # removes any words with 3 or more repeated letters
#     !str_detect(word2, pattern = "(.)\\1{2,}"),
#     !str_detect(word1, pattern = "\\b(.)\\b"), # removes any remaining single letter words
#     !str_detect(word1, pattern = "\\b(.)\\b")
#   ) %>%
#   unite("bigram", c(word1, word2), sep = " ") %>%
#   count(bigram) %>%
#   filter(n >= 10) %>% # filter for bi-grams used 10 or more times
#   pull(bigram)

n_gram_list <- txt_c %>%
  select(c("text", "labels", "doc_id")) %>%
  unnest_tokens(output = bigram, input = text, token = "ngrams", n =2) %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(
    !word1 %in% stop_words$word, # remove stopwords from both words in bi-gram
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
  unite("bigram", c(word1, word2), sep = " ") %>%
  count(bigram) %>%
  filter(n >= 5) %>% # filter for bi-grams used 10 or more times
  pull(bigram)

# sneak peek at our bi-gram list
head(n_gram_list)
length(n_gram_list)
# Adding 151 new features 

# create new bi-gram features
n_gram_list_2 <- txt_c %>%
  select(c("text", "labels", "doc_id")) %>% 
  unnest_tokens(word, input=text, token = "ngrams", n = 2) %>%
  filter(word %in% n_gram_list) # filter for only bi-grams in the ngram_list
  # count(doc_id, bigram) %>% # count bi-gram useage by doc ID
  # spread(bigram, n) %>% # convert to wide format
  # map_df(replace_na, 0) # replace NAs with 0

# create ngram - 1 features
n_gram_list_1 <- txt_c %>%
  select(c("text", "labels", "doc_id")) %>% 
  unnest_tokens(output = word, input = text)

n_gram_list_1_no_sw <- n_gram_list_1 %>%
  anti_join(get_stopwords()) %>%
  filter(
    !str_detect(word, pattern = "[[:digit:]]"), # removes any words with numeric digits
    !str_detect(word, pattern = "[[:punct:]]"), # removes any remaining punctuations
    !str_detect(word, pattern = "(.)\\1{2,}"), # removes any words with 3 or more repeated letters
    !str_detect(word, pattern = "\\b(.)\\b"), # removes any remaining single letter words
  ) 

# Complete N-gram list with the bigrams and ngrams
full_n_gram_list <- rbind(n_gram_list_1_no_sw, n_gram_list_2) 

df_full <- full_n_gram_list %>% 
  count(doc_id, word) %>% # count bi-gram useage by doc ID
  spread(word, n) %>% # convert to wide format
  map_df(replace_na, 0) %>%
  add_column(labels = c(rep(0,320), rep(1,320)))

# DECEPTIVE == 0, TRUTHFULL == 1


# Create dtm: 320 deceptive, 320 truthful 
class_dtm_full_n_gram_list <- full_n_gram_list %>%
  count(doc_id, word) %>%
  cast_dtm(document = doc_id, term = word, value = n)

# Reducing model complexity by removing sparse terms from the model: tokens that do not appear across many documents 
class_dtm_full_2<- removeSparseTerms(class_dtm_full_n_gram_list, sparse =.99) #97 terms 562


### FOR THE TEST SET 
# create new bi-gram features
test_set_bigram_features <- txt_test %>%
  select(c("text", "labels", "doc_id")) %>% 
  unnest_tokens(word, input=text, token = "ngrams", n = 2)

# create ngram - 1 features
test_set_1 <- txt_test %>%
  select(c("text", "labels", "doc_id")) %>% 
  unnest_tokens(output = word, input = text)

# Complete N-gram list with the bigrams and ngrams
full_test_set <- rbind(test_set_1, test_set_bigram_features) 

# Create dtm: 80 deceptive, 80 truthful 
test_dtm <- full_test_set %>%
  count(doc_id, word) %>%
  cast_dtm(document = doc_id, term = word, value = n)

test_dtm

###### COMBINE TRAINING AND TEST 
### 640 (truthful, deceptive) - 180 test (deceptive, truthful)
txt_full <- rbind(txt_c,txt_test)

n_gram_list_full_2 <- txt_full %>%
  select(c("text", "labels", "doc_id")) %>%
  unnest_tokens(output = bigram, input = text, token = "ngrams", n =2) %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(
    !str_detect(word1, pattern = "[[:digit:]]"), # removes any words with numeric digits
    !str_detect(word2, pattern = "[[:digit:]]"),
    !str_detect(word1, pattern = "[[:punct:]]"), # removes any remaining punctuations
    !str_detect(word2, pattern = "[[:punct:]]"),
    !str_detect(word1, pattern = "(.)\\1{2,}"), # removes any words with 3 or more repeated letters
    !str_detect(word2, pattern = "(.)\\1{2,}"),
    !str_detect(word1, pattern = "\\b(.)\\b"), # removes any remaining single letter words
    !str_detect(word1, pattern = "\\b(.)\\b")
  ) %>%
  unite("bigram", c(word1, word2), sep = " ") %>%
  count(bigram) %>%
  pull(bigram)

# sneak peek at our bi-gram list
head(n_gram_list_full_2)
length(n_gram_list_full_2)

# Adding 51596 new features 
n_gram_list_full_2b <- txt_full %>%
  select(c("text", "labels", "doc_id")) %>% 
  unnest_tokens(word, input=text, token = "ngrams", n = 2) %>%
  filter(word %in% n_gram_list_full_2) # filter for only bi-grams in the ngram_list
# count(doc_id, bigram) %>% # count bi-gram useage by doc ID
# spread(bigram, n) %>% # convert to wide format
# map_df(replace_na, 0) # replace NAs with 0

# create ngram - 1 features
n_gram_list_full_1 <- txt_full %>%
  select(c("text", "labels", "doc_id")) %>% 
  unnest_tokens(output = word, input = text) %>%
  filter(
    !str_detect(word, pattern = "[[:digit:]]"), # removes any words with numeric digits
    !str_detect(word, pattern = "[[:punct:]]"), # removes any remaining punctuations
    !str_detect(word, pattern = "(.)\\1{2,}"), # removes any words with 3 or more repeated letters
    !str_detect(word, pattern = "\\b(.)\\b"), # removes any remaining single letter words
  ) 

# Complete N-gram list with the bigrams and ngrams
full_set <- rbind(n_gram_list_full_1, n_gram_list_full_2b)

dtm_full <- full_set %>%
    count(doc_id, word) %>%
    cast_dtm(document = doc_id, term = word, value = n)
  

# dtm_full_test <- full_set %>%
#   filter(doc_id %in% txt_test$doc_id) %>%
#   count(doc_id, word) %>%
#   cast_dtm(document = doc_id, term = word, value = n)

# df_full <- full_set %>% 
#   count(doc_id, word) %>% # count bi-gram useage by doc ID
#   spread(word, n) %>% # convert to wide format
#   map_df(replace_na, 0) 

# df_full_training <- full_set %>%
#   filter(!doc_id %in% txt_test$doc_id) %>%
#   add_column(labels = c(rep(0,320), rep(1,320)))

# dtm_full_training <-  full_set %>%
#   filter(!doc_id %in% txt_test$doc_id) %>%
#   count(doc_id, word) %>%
#   cast_dtm(document = doc_id, term = word, value = n)

dtm_full_training
# DECEPTIVE == 0, TRUTHFULL == 1

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

# load test data

corpus_test <- Corpus(DirSource("./test"), readerControl = list(language="lat"))

corpus_test <- tm_map(corpus_test, removeNumbers)
corpus_test <- tm_map(corpus_test, removePunctuation)
corpus_test <- tm_map(corpus_test , stripWhitespace)
corpus_test <- tm_map(corpus_test, tolower)
corpus_test <- tm_map(corpus_test, removeWords, stopwords("english"))
corpus_test <- tm_map(corpus_test, stemDocument)

dtm_test <- DocumentTermMatrix(corpus_test)
sparse_test <- as.matrix(dtm_test)

sparse_train <- as.matrix(sparse)

# Multinominal naive bayes:
naive_bayes <- train.mnb(dtm = class_dtm_c,labels=c('deceptive','truthful')) 

prediction <- predict.mnb(model=naive_bayes, dtm = test_dtm)

### LOGISTIC REGRESSION
dtm_full_tr <- df_full_training %>%
  count(doc_id, word) %>%
  cast_dtm(document = doc_id, term = word, value = n)

dtm_test <- dtm_full[dtm_full$dimnames$Docs %in% txt_test$doc_id,]
dtm_training <- dtm_full[!dtm_full$dimnames$Docs %in% txt_test$doc_id,]

model <- glmnet(x = as.matrix(dtm_training),y = c(rep(0, 320), rep(1, 320)), family = "binomial")

# rlr <- glmnet(x = as.matrix(sparse), y = c(rep(FALSE, 320), rep(TRUE, 320)), family = "binomial", nlambda=100)
test <- predict(model, newx = as.matrix(dtm_test), type = "response")


# # Regularized logistic regression:
# # x: input matrix of dimension nr. obs. * nr. vars. (each row one observation)
# # y: response variable
# # (https://glmnet.stanford.edu/articles/glmnet.html#logistic-regression)
# # set.seed(421)
# rlr <- glmnet(x = as.matrix(sparse), y = c(rep(FALSE, 320), rep(TRUE, 320)), family = "binomial", nlambda=100)
# plot(rlr)
# 
# # predict(rlr, newx=sparse_train, type="response")
# 
# lrlr <- glmnet(x = as.matrix(class_dtm_full_2), y = c(rep("deceptive", 320), rep("truthfull", 320)), family = "binomial")
# predict(lrlr, newx = as.matrix(test_dtm), type= "response")
# 
# cvrlr<- cv.glmnet(x = as.matrix(sparse), y = c(rep(FALSE, 320), rep(TRUE, 320)), family = "binomial", type.measure = "class")
# plot(cvrlr)
# predict(cvrlr, newx = as.matrix(dtm_test), type="response")
# 
# cvrlr$lambda.min
# cvrlr$lambda.1se
# 
# # Final model with lambda.min
# lasso.model <- glmnet(x = as.matrix(sparse), y = c(rep(FALSE, 320), rep(TRUE, 320)), alpha = 1, family = "binomial",
#                       lambda = cvrlr$lambda.min)
# 
# # Make prediction on test data
# probabilities <- predict(lasso.model, newx = sparse_test, type="response") # newx = test-data
# predicted.classes <- ifelse(probabilities > 0.5, "truthful", "deceptive") 

# Model accuracy
observed.classes <- test_data$labels
mean(predicted.classes == observed.classes)


# function for accuracy, precision, and F1-score
confusionmatrix <- function(true, pred) {
  conf_matrix <- table(true = true, pred = pred)
  
  TP <- conf_matrix[1,1]
  FP <- conf_matrix[1,2]
  FN <- conf_matrix[2,1]
  TN <- conf_matrix[2,2]
  
  accuracy <- (TP + TN) / (TP + TN + FP + FN)
  
  precision <- (TP) / (TP + FP)
  
  F1 <- (2 * TP) / (2 * TP + FP + FN)
}

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
# 
# ###
# # n_gram_list_2 = 558 X 150 (Thus: 150 features)
# # n_gram_list_1_no_sw = 640 x 6,525 (Thus: 6.525 features)
# # full n_gram_list =  640 x 6,675:
# full_n_gram <- left_join(n_gram_list_1_no_sw, n_gram_list_2, by = "doc_id") 
# 
# # add column into deceptive / truthful, drop the "doc_id"
# full_n_gram_2 <- left_join(txt_c %>% select(c("labels","doc_id")),
#                            full_n_gram , by = "doc_id") %>%
#   select(c(-"doc_id"))
# 
# # So, now you have a full n-gram df: every row is either truthful or deceptive with their 
# # own features, either one word, or bigrams
# head(full_n_gram_2)
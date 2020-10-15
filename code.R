library(rpart.plot)
library(tm)
library(SnowballC)
library(randomForest)
library(tidyverse)
library(quanteda)
library(tidytext)
library(qdap)
library(ggplot2)

# Create corpus
# Negative deceptive 
corpus_d <- Corpus(DirSource("./deceptive/allfortm_d"), readerControl = list(language="lat")) #specifies the exact folder where my text file(s) is for analysis with tm.
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
labels <- c(rep("truthfull",320), rep("deceptive",320))

# Complete text with labels [640x3]
txt_c <- txt_c %>%
  add_column(labels = factor(labels)) 

stopwords <- tm::stopwords(kind = "en")

# Prepare for word_cloud (no further preprocessing: needs to remove stopwords)
dfm_corpus_t <- txt_c[txt_c$labels == "truthfull",] %>%
  corpus(text_field = "text") %>%
  tokens(remove_numbers = FALSE, remove_punct = TRUE, remove_symbols = TRUE, remove_separators = TRUE, split_hyphens	
         = TRUE, remove_url = TRUE) %>% 
  dfm()

dfm_corpus_d <- txt_c[txt_c$labels == "deceptive",] %>%
  corpus(text_field = "text") %>%
  tokens(remove_numbers = FALSE, remove_punct = TRUE, remove_symbols = TRUE, remove_separators = TRUE, split_hyphens	
 = TRUE, remove_url = TRUE) %>%
  dfm() %>%
  removeSparseTerms(0.995)

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

# Create dtm
class_dtm_c <- cleaned_data_tokens_nosw %>%
  count(doc_id, word) %>%
  cast_dtm(document = doc_id, term = word, value = n)

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

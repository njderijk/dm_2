Planning voor A2 (Deadline: 30 oktober)

Taken:

1. Multinomial naive Bayes (generative linear classifier) & feature selection
2. Regularized logistic regression (discriminative linear classifier) & feature selection
3. Classification trees, (flexible classifier) & feature selection
4. Random forests (flexible classifier) & out-of-bag evaluation

640 reviews for training & hyper-parameter tuning.
160 reviews for testing.


Performance measures: accuracy, precision, recall, and F1 score.


Volgende vragen in het verslag beantwoorden:
1. How does the performance of the generative linear model (multinomial
naive Bayes) compare to the discriminative linear model (regularized
logistic regression)?
2. Is the random forest able to improve on the performance of the linear
classifiers?
3. Does performance improve by adding bigram features, instead of using
just unigrams?
4. What are the five most important terms (features) pointing towards a
fake review?
5. What are the five most important terms (features) pointing towards a
genuine review?


* Planning

16 oktober
* Kijken volgens welke template het verslag opgesteld moet worden
* Introduction (Yasmin)
* Data section (Noud)
* Verder met initiële code (Rose) & alvast packages op een rijtje zetten die gebruikt kunnen worden


21 oktober
* Exploratory Data Analysis
  * Word clouds with pre-processed terms (Noud)
  * Text length / count of words (truthful vs. deceptive) (Yasmin)
  * Top 20 words before preprocessing (Noud)
  * Top 20 words after pre-processing (Noud)
  * Top 20 bigrams (n-gram analysis) (Yasmin)
  * Sentiment analysis (Noud)
  * tf-idf (Yasmin)
* Gelijktijdig met EDA tekst schrijven (Iedereen)


23 oktober
* Exploratory Data Analysis
  * Teksten schrijven bij figuren en uitleggen wat het toevoegt (Yasmin & Noud)
  * Stopwoorden uit figuren halen en aanpassen (Noud)
  * Truthful en deceptive reviews apart voor o.a. N-grams (Noud)
  * tf-idf splitsen in truthful en deceptive (Yasmin)
  
* Eerste analyses zonder feature selection
  * Classifications en random forests (Noud)
  * Multinomial naive bayes (Rose)
  * Regularized logistic regression (Yasmin)
  * Template opstellen voor accuracy, precision, en F1-score (Yasmin)
  * Uitdenken cross-validation / feature selection (Iedereen)
  * Uitdenken hoe je bigrams toevoegt en hoe je erop traint (Noud)


28 oktober (of eerder)
 * Vragen in Teams over cross-validation van multinomial naive bayes
 
 * Exploratory Data Analysis
  * tf-idf top 10 i.p.v. top 50 (Yasmin)
  
 * Test data inladen om op te predicten
  
 * Methoden
  * Voorspellen met bigrams met analyses/modellen (Rose)
  * Doorgaan met analyses / modellen maken (Iedereen, zie hieronder)
    * Regularized logistic regression (Rose)
    * Multinomial naive bayes (Yasmin)
    * Classifcation Tree (Noud)
    * Random Forest (Noud)
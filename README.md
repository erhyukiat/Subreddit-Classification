# Subreddit Classification

Problem Statement
---
The "TheOnion" subreddit is a subreddit where users can post news articles found on https://www.theonion.com, which is a satirical news site. It is fairly well known that The Onion articles are all satirical in nature, so the news titles usually sound ridiculous. The "nottheonion" subreddit is a subreddit where users can post news articles of true stories that are so mind-blowingly ridiculous that you could have sworn they were from The Onion.

When performing maintenance, an engineer accidentally deleted multiple posts from r/nottheonion and r/theonion. Unfortunately, the engineer was only able to recover the titles of the lost posts. We were therefore tasked to build a classification model which would train on posts submitted before 2022 to classify the recovered posts back to their respective subreddits, r/nottheonion and r/theonion, based solely on the post titles.

Reddit has been thinking of developing machine learning models for automoderators, therefore, we will also be making use of this opportunity to do a proof of concept that machine learning (ML) models could be used to develop automoderators. Moderators are currently spending a substantial amount of their time reviewing user reports and deleting irrelevant posts from the subreddit. In theory, if automoderators developed through ML were deployed, it would be able to classify whether a posts belongs in the subreddit it is in or not. Having automoderators police the subreddit for irrelevant posts would free up time for human moderators, who are volunteers, to do things that they want to do.

We will explore using four models (Random Forest Classifier, Logistic Regression, Multinomial Naive Bayes and Support Vector Classifier) and select the model that is the most successful in classifying the subreddit posts. Success will be evaluated through the accuracy score and F1 score, as we want both false positives and false negatives to be minimised.

Executive Summary
---
Data was collected from two subreddits, r/TheOnion and r/nottheonion, using the PushShift API. A total of 4000 posts were collected, 2000 from each subreddit. As it is possible that the same post could be posted multiple times to farm Reddit karma, we ensured that no duplicate posts or reposts were collected by searching through the list of posts scraped at that point in time. After dropping data fields that seemed irrelevant during the data collection step, the data was loaded into a data frame, and analysis was performed on the titles of the posts using Natural Language Processing techniques. 

The data was found to have no missing values, which is expected since the data comes from Reddit. The average post lengths in terms of both character lengths and word lengths were similar between the two subreddits. The title text was converted into a bag of words using CountVectorizer for further analysis. The most popular unigrams and bigrams for r/TheOnion indicated that satirical article titles are up-to-date with current affairs, as they mention US politics and the global pandemic. As for r/nottheonion, it indicated that the ridiculous news usually involved a person, and more often than not, a "Florida man".

The title text of the posts were preprocessed by removing stopwords from the nltk library, and lemmatized using the WordNetLemmatizer, before it was fed into a pipeline containing TfidfVectorizer and a classification model. A total of 4 classification models were explored: Random Forest, Logistic Regression, Multinomial Naive Bayes and Support Vector Machine. GridSearchCV was used to tune the hyperparameters of the TfidfVectorizer and the classification models. A 75/25 train test split was performed before feeding the data into each pipeline. Each model was evaluated using the accuracy score and F1 score. The confusion matrix and ROC curves were also analysed to see if the findings matched the accuracy and F1 scores. The best performing model was found to be the Support Vector Classifier with an accuracy of 79%. Given that this is the first iteration of a possible ML automoderator, the results are promising and show that ML automoderators are feasible.

Data Dictionary
---
| Field       | Description                          |
|-------------|--------------------------------------|
| author      | Author of the post                   |
| created_utc | Unix time of post creation           |
| domain      | Domain of the URL linked in the post |
| full_link   | Reddit URL to the post               |
| id          | Unique id of the post                |
| subreddit   | Subreddit of the post                |
| title       | Title of the post                    |
| url         | URL linked in the post               |

Model Scores
---
|                                   | Train accuracy | Test accuracy | Cross val score | F1 score |
|-----------------------------------|----------------|---------------|-----------------|----------|
| Baseline model (Dummy Classifier) | 0.508          | 0.476         | 0.508           | -        |
| Random Forest Classifier          | 0.995          | 0.747         | 0.712           | 0.749    |
| Logistic Regression               | 0.996          | 0.776         | 0.732           | 0.780    |
| Multinomial Naive Bayes           | 0.995          | 0.779         | 0.740           | 0.777    |
| Support Vector Classifier         | 0.999          | 0.786         | 0.742           | 0.797    |

Conclusion & Recommendation
---
We were tasked to build classification model that could classify whether a post belonged to r/TheOnion or r/nottheonion based solely on the post titles, given that only the post titles were recovered from the deleted data. The best performing model was a Support Vector Classifier model which had features created using the TfidfVectorizer. TfidfVectorizer is preferred over CountVectorizer as it got better results when loading the vectorized data into the models, and also it gives the different words a different weight based on how often it appears whereas CountVectorizer simply counts the number of occurrences which may not be as useful. The model was able to predict with an accuracy of 78.6% and has a F1-score of 79.7%.

Hence, with an accuracy score of almost 80%, we are confident of correctly classifying 7-8 out of every 10 deleted posts. We therefore recommend using the Support Vector Classifier with TfidfVectorizer to classify the recovered deleted posts. For the posts that were wrongly classified, we would have to rely on user reports and/or manual verification by the moderators of the respective subreddits. The results also show that classification models would be feasible to be used in the development of automated moderators.

To improve the model, more data can be gathered to train the model since machine learning models can always benefit from more data. Another thing we could do is to examine the text more closely, to remove more words as stopwords. We could also look into engineering more features using the metadata of reddit posts, like word count, character count, sentiment analysis etc.

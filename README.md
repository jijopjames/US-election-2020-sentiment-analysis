# US-election-2020-sentiment-analysis

Predicting US Presidential Election 2020 Result Using Twitter Sentiment Analysis with Python

## Dataset creation

- **Using twitter API to scrape tweets**
  - Copy “API Key”, “API Secret”, “Access Token”, and “Access Token Secret” to use as Oauth keys.
  - Setup Authentication with Twitter using tweepy package.
  - Extracting tweets for both Donald Trump and Joe Biden.

  ```python
  candidate_name = ['realDonaldTrump','JoeBiden']
  
  replies_trump = []
  replies_biden = []
  for candidate in candidate_name:
      for tweet in tweepy.Cursor(api.search,q='to:'+candidate, result_type='recent',timeout=999999).items(10000):
          if candidate == "realDonaldTrump":
              replies_trump.append(tweet)
            
          elif candidate == "JoeBiden":'
              replies_biden.append(tweet)
  ```

  - converting the files into dataframe and to csv
  
  ```python
  biden_df = pd.DataFrame()
  trump_df = pd.DataFrame()
  
  df_names = ['biden_df','trump_df']

  for tweet in replies_trump:
      row = {'user': tweet.user.screen_name, 'text': tweet.text.replace('\n', ' ')}
      trump_df = trump_df.append(row, ignore_index=True)
      trump_df.to_csv(r'data/trump_data.csv')
    
  for tweet in replies_biden:
      row = {'user': tweet.user.screen_name, 'text': tweet.text.replace('\n', ' ')}
      biden_df = biden_df.append(row, ignore_index=True)
      biden_df.to_csv(r'data/biden_data.csv')
  ```
  
  - generated data is avaliabe in 'trump_data.csv' and 'biden_data.csv'.
  - both the dataset contains to 2 columns and 10000 rows.
    - 'text' - this coloumn contains tweets containing '@realDonaldTrump' or '@JoeBiden' respectively.
    - 'user' - this coloumn contains the username. 
  

## Data Analysis

- **Importing the datasets**
- **Sentiment analysis using TextBlob**
  - Polarity ranges from -1 to +1 and tells whether the text has negative sentiments or positive sentiments
  - polarity function returns the polarity of each tweet
  
  ```python
  def polarity(review):
    return TextBlob(review).sentiment.polarity
    
    Trump_reviews['polarity'] = Trump_reviews['text'].apply(polarity)
    Biden_reviews['polarity'] = Biden_reviews['text'].apply(polarity) 
  ```
  
  - adding the tag of 'Positive', 'Negative' or 'Netural' according to the polarity
  
  ```python
  Trump_reviews['Expression'] = np.where(Trump_reviews['polarity']>0,'Positive','Negative')
  Trump_reviews.loc[Trump_reviews.polarity == 0, 'Expression'] = 'Netural'
  Trump_reviews.head()
  
  Biden_reviews['Expression'] = np.where(Biden_reviews['polarity']>0,'Positive','Negative')
  Biden_reviews.loc[Biden_reviews.polarity == 0, 'Expression'] = 'Netural'
  Biden_reviews.head()
  ```
  - Visualizing to find Positive, Negative and Neutral
  
  ![Trump_review_analysis](https://github.com/jijopjames/US-election-2020-sentiment-analysis/blob/master/img/trump_review.png)
  ![Biden_review_analysis](https://github.com/jijopjames/US-election-2020-sentiment-analysis/blob/master/img/biden_review.png)
  
  - Droping all neutral data since they do not add value to the analysis
  ```python
  Trump_reviews.drop((Trump_reviews[Trump_reviews['polarity']==0]).index, inplace=True)
  print(Trump_reviews.shape)
  Biden_reviews.drop((Biden_reviews[Biden_reviews['polarity']==0]).index, inplace=True)
  print(Biden_reviews.shape)
  ```
  - After droping the neutral data the I have an uneven dataset to balance out both datasets I make use of 'balanced_data' function.
  ```python
  def balanced_data(reviews,n):
    np.random.seed(10)
    drop = np.random.choice(reviews.index,n,replace=False)
    review_subset = reviews.drop(drop)
    return review_subset
  
  Trump_subset = balanced_data(Trump_reviews,99)
  print(Trump_subset.shape)

  Biden_subset = balanced_data(Biden_reviews,300)
  print(Biden_subset.shape)
  ```
  - After balancing the data we have 4000 rows in each dataset.
 
## Data Visualization

- **Donald Trump** 
  - From the below figure, one can easily interpret that polarity ranges from -1 to +1 and a larger number of people have positive reviews because it is mostly concentrated between 0 and 0.5.
  
  ![distplot_trump](https://github.com/jijopjames/US-election-2020-sentiment-analysis/blob/master/img/trump_distplot.png)
  
  - From below figure of boxplot, one can easily identify most of the polarity is concentrated between -0.25 to 0.50. So, it is basically showing only the concentration of polarity.

  ![boxplot_trump](  https://github.com/jijopjames/US-election-2020-sentiment-analysis/blob/master/img/trump_boxplot.png)
  
  - Analyzing Most Positive and Most Negative replies
    - **Note:-** As per the insights I have gained by this project. 'TextBlob sentiment analyzer' is not efficient enough to detect the scarcastic comments. Since, it works on tokens of sentence and classify accordingly. 
    
    ![most_pos_trump](https://github.com/jijopjames/US-election-2020-sentiment-analysis/blob/master/img/trump_pos_tweet.png) 
    
    ![most_neg_trump](https://github.com/jijopjames/US-election-2020-sentiment-analysis/blob/master/img/trump_neg_tweet.png)
    
   - Word clouds can be useful to find your customer's pain points in business purposes, I am using it to get insights of public opinion about the presidential candidate and most frequently used keywords by the citizens.
   
   ![trump_wordcloud](https://github.com/jijopjames/US-election-2020-sentiment-analysis/blob/master/img/trump_wordcloud.png) 
   
- **Joe Biden**
  - From the below figure, one can easily interpret that polarity ranges from -1 to +1 and a larger number of people have positive reviews because it is mostly concentrated between 0 and 0.5.
  
  ![distplot_biden](https://github.com/jijopjames/US-election-2020-sentiment-analysis/blob/master/img/biden_distplot.png)
  
  - From below figure of boxplot, one can easily identify most of the polarity is concentrated between -0.25 to 0.50. So, it is basically showing only the concentration of polarity.

  ![boxplot_biden](  https://github.com/jijopjames/US-election-2020-sentiment-analysis/blob/master/img/biden_boxplot.png)
  
  - Analyzing Most Positive and Most Negative replies
    - **Note:-** As per the insights I have gained by this project. 'TextBlob sentiment analyzer' is not efficient enough to detect the scarcastic comments. Since, it works on tokens of sentence and classify accordingly. 
    
    ![most_pos_biden](https://github.com/jijopjames/US-election-2020-sentiment-analysis/blob/master/img/biden_pos_tweet.png) 
    
    ![most_neg_biden](https://github.com/jijopjames/US-election-2020-sentiment-analysis/blob/master/img/biden_neg_tweet.png)
    
   - Word clouds can be useful to find your customer's pain points in business purposes, I am using it to get insights of public opinion about the presidential candidate and most frequently used keywords by the citizens.
   
   ![biden_wordcloud](https://github.com/jijopjames/US-election-2020-sentiment-analysis/blob/master/img/biden_wordcloud.png) 

- **People Sentiment**

  -  From the below figures, it is very evident that Joe Biden is getting more positive replies as compare to negative reviews.
  
  ![candidate_pos_neg](https://github.com/jijopjames/US-election-2020-sentiment-analysis/blob/master/img/candidate_pos_neg.png) 
  
  - The overall people sentiment is more favouralbe to Joe Biden over Donald Trump.
    - **Note:-** I am assuming the all the users are unique. Hence, I have note removed the users who commented on both Joe Biden & Donald Trump
  
  ![people_sentiment](https://github.com/jijopjames/US-election-2020-sentiment-analysis/blob/master/img/people_sentiment.png) 
  
  



  

 




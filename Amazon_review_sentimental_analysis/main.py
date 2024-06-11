import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import warnings

nltk.download('vader_lexicon')
warnings.filterwarnings('ignore')

# Load the dataset
def load_dataset():
    data = pd.read_csv('Amazon_review_sentimental_analysis/Reviews.csv')
    return data

# Drop the missing columns and reset index
def pre_processing():
    data = load_dataset().dropna().reset_index(drop=True)
    return data

# Finding the distribution of ratings
def distribution_of_ratings():
    data = pre_processing()
    ratings = data['Score'].value_counts()
    return ratings

# Calculating positive, negative and neutral values in the dataframe and merging with the original dataframe
def add_new_columns():
    data = pre_processing()
    analyzer = SentimentIntensityAnalyzer()

    pos = []
    neg = []
    neu = []

    for i in range(len(data)):
        scores = analyzer.polarity_scores(data.loc[i, 'Text'])
        pos.append(scores['pos'])
        neg.append(scores['neg'])
        neu.append(scores['neu'])
    
    data['Positive'] = pos
    data['Negative'] = neg
    data['Neutral'] = neu

    data = data[['Id','ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator','Score','Time','Summary','Text','Positive','Negative','Neutral']]
    return data

# Finding the sum of positive, negative and neutral values
def sentiment_scores():
    data = add_new_columns()

    pos_sum = data['Positive'].sum()
    neg_sum = data['Negative'].sum()
    neu_sum = data['Neutral'].sum()

    return pos_sum, neg_sum, neu_sum

# Calling all functions
data = pre_processing()
print(data)

ratings = distribution_of_ratings()
print('The rating distribution is\n', ratings)

x, y, z = sentiment_scores()
print(f'Positive Score: {x}, Negative Score: {y}, Neutral Score: {z}')

if max(x, y, z) == x:
    print('The sentiment with the maximum score is Positive')
elif max(x, y, z) == y:
    print('The sentiment with the maximum score is Negative')
else:
    print('The sentiment with the maximum score is Neutral')

# Natural Language Processing (NLP) Projects

### 1) Amazon Reviews Sentiment Analysis
**Description**: This project involves sentiment analysis of approximately 15,000 user reviews from Amazon, a leading global e-commerce platform.

**Methodology**: Similar to the Flipkart project, I utilized NLTK for sentiment analysis to classify the reviews into positive, negative, and neutral sentiments.

**Results**: The sentiment scores for the Amazon reviews are:
- **Positive Score**: 2879.631
- **Negative Score**: 639.59
- **Neutral Score**: 11480.747

These scores highlight the general customer sentiment towards products available on Amazon, aiding in understanding consumer behavior and satisfaction.

### 2) COVID-19 Tweet Classification
**Description**: This project centers around the sentiment analysis of tweets related to COVID-19, utilizing a dataset from Kaggle that includes 41,157 training tweets and 3,798 test tweets, categorized into five sentiment types: Extremely Negative, Extremely Positive, Negative, Neutral, and Positive.

**Methodology**:
- **Initial Approach**: Employed the Multinomial Naive Bayes algorithm with grid search hyperparameter tuning, achieving an initial accuracy of 48.05% with 
- Macro Average: Precision: 0.49, Recall: 0.49, F1-Score: 0.49
- Weighted Average: Precision: 0.48, Recall: 0.48, F1-Score: 0.48
- **Enhanced Approach**: Transitioned to using BERT (bert-base-uncased) for sequence classification. The model was fine-tuned over 10 epochs, with optimized tokenization and attention masks.

**Results**: The BERT model significantly improved the accuracy to 83.73% on the test data, highlighting the effectiveness of transfer learning and contextual embeddings in NLP tasks.

### 3) Flipkart Reviews Sentiment Analysis
**Description**: This project focuses on analyzing user reviews from Flipkart, one of India's largest e-commerce platforms. The dataset comprises approximately 2,300 reviews written by customers about various products.

**Methodology**: Using the Natural Language Toolkit (NLTK), I performed sentiment analysis to determine the emotional tone of the reviews. The analysis categorizes each review into positive, negative, or neutral sentiments.

**Results**: The sentiment scores for the reviews are:
- **Positive Score**: 835.67
- **Negative Score**: 104.917
- **Neutral Score**: 1363.413

These scores provide insights into customer satisfaction and the overall perception of the products reviewed on Flipkart.

### 4) Language Prediction
**Description**: This project involves predicting the language of text from a dataset containing 22 languages, each with 1,000 records.

**Methodology**: I used a count vectorizer to preprocess the data and the Multinomial Naive Bayes algorithm to build the model, as it is well-suited for multi-class classification problems.

**Results**: The model achieved an accuracy of 95.316%, demonstrating its capability to accurately identify the language of the given text data.

### 5) Spam Mail Detection
**Description**: This project focuses on identifying spam emails from a dataset of 5,500 email messages classified as either spam or ham (non-spam).

**Methodology**: I performed extensive data preprocessing and built a predictive model to classify the emails. The Bernoulli Naive Bayes algorithm was found to be the most effective for this task.

**Results**: The model achieved a precision and accuracy of 0.97, demonstrating its high effectiveness in distinguishing between spam and ham emails. This project underscores the potential of machine learning in enhancing email filtering systems.
- The GaussianNB model accuracy and precision are 0.876 and 0.523
- The MultinomialNB model accuracy and precision are 0.959 and 1.0
- The BernoulliNB model accuracy and precision are 0.970 and 0.973

**Usage**: To run the code for sentiment analysis on Flipkart and Amazon reviews, as well as the spam mail detection, install the NLTK library and use the command `python main.py`.

### 6) Text Summarization
**Description**: This project defines functions for generating abstractive summaries of texts using models from Hugging Face's Transformers library.

**Methodology**:
- **Model and Tokenizer Loading**: The `load_model_and_tokenizer` function retrieves a specified pre-trained model and its tokenizer.
- **Summary Generation**: The `abstractive_summary` function takes in a text, the desired model name, and optional arguments for input and output length. It uses the model to generate a summarized version of the input text.

**Usage**: When executed, the script prompts the user to select a model from options like `t5-small`, `facebook/bart-large-cnn`, etc., and then input the text they want to summarize. After processing, the script outputs the generated summary to the console.

These projects showcase the diverse applications of NLP techniques, from sentiment analysis and spam detection to language prediction and text summarization, demonstrating the potential of machine learning and deep learning in understanding and processing human language.


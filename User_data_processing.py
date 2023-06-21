import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import ngrams
from nltk.corpus import words
import os
import numpy
import csv
from textblob import TextBlob
import textdistance
import collections
import re  # regular expression
import pandas as pd


data=input('please enter the data')


def text_to_lower(data):
    data_lower_text = data.lower()
    return data_lower_text


def remove_punctuation(data):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    news_no_punct = ""
    for i in data:
        if i not in punctuations:
            news_no_punct = news_no_punct + i
    news_no_punct = news_no_punct.strip()

    return news_no_punct


def remove_single_characters(data):
    new_text = ""
    tokenized = nltk.tokenize.word_tokenize(data)
    for w in tokenized:
        # print(w)
        if len(w) > 1:
            if len(new_text) > 0:
                new_text = new_text + " " + w
            else:
                new_text = w
    return new_text


def remove_stop_words(data):
    stop_words = set(stopwords.words('english'))
    tokenized = nltk.tokenize.word_tokenize(data)
    new_text = ""
    for word in tokenized:
        if word not in stop_words and word.isalpha():
            new_text = new_text + " " + word
    return new_text


def count_words(data):
    tokenized = nltk.tokenize.word_tokenize(data)
    total_words = len(tokenized)
    return total_words


def compute_unique_words(data):
    token_words = []
    uniqueWords = []
    stop_words = set(stopwords.words('english'))
    tokenized = nltk.tokenize.word_tokenize(data)
    for i in tokenized:
        if i not in stop_words and i.isalpha():
            token_words.append(i)
    clean_token_words = remove_punctuation(token_words)
    for w in token_words:
        if w not in uniqueWords:
            uniqueWords.append(w)
    return len(uniqueWords)


def is_english_word(data):
    counter_flag = False
    uniqueWords = []
    token_words = []
    data = text_to_lower(data)

    data = remove_punctuation(data)

    stop_words = set(stopwords.words('english'))

    tokenized = nltk.tokenize.word_tokenize(data)
    for i in tokenized:
        if i not in stop_words and i.isalpha():
            token_words.append(i)
    for w in token_words:
        if w not in uniqueWords and w in words.words():
            counter_flag = True
            uniqueWords.append(w)
            break
    return counter_flag


def compute_total_characters(data):
    tokenized = nltk.tokenize.word_tokenize(data)
    total_chars = 0
    for w in tokenized:
        for c in w:
            if c.isspace() != True:
                total_chars += 1
    return total_chars;





def compute_sentiment_score(review_score ,review_text,app_title,app_description):
    stop_words = set(stopwords.words('english'))
    token_words = []
    positive_words_list = []
    negative_words_list = []
    positive_words_polarity = []
    negative_words_polarity = []
    total_positive_polarity = 0.0
    total_negative_polarity = 0.0
    p_ratio = 0.0
    n_ratio = 0.0
    f = [None] * 15;
    p_n_label ='none'



    # Text to lower
    review_text_lower =text_to_lower(review_text)
    # Total words count with Punctuation
    total_count_with_punch =count_words(review_text_lower)
    # Total characters count with Punctuation and Stop words
    total_chars_count_with_punch =compute_total_characters(review_text_lower)
    # Remove Punctuation
    review_text_clean_punt =remove_punctuation(review_text_lower)
    # Remove Stop words
    text_without_stopwords =remove_stop_words(review_text_lower)
    # Total words without Punctuation and Stop words
    total_count_without_punch =count_words(text_without_stopwords)
    # Total characters count with Punctuation and Stop words
    total_chars_count_without_punch = compute_total_characters(text_without_stopwords)
    # Total number of unique words
    unique_words_count = compute_unique_words(review_text_lower)

    # Clean Title content




    tokenized = nltk.tokenize.word_tokenize(review_text_lower)

    for i in tokenized:
        # Word tokenizers is used to find the words and punctuation in a string
        wordsList = nltk.word_tokenize(i)
        if i not in stop_words and i.isalpha():
            token_words.append(i)
    for w in token_words:
        blob = TextBlob(w)
        if blob.polarity > 0:
            positive_words_list.append(w)
            positive_words_polarity.append(blob.polarity)
        elif blob.polarity < 0:
            negative_words_list.append(w)
            negative_words_polarity.append(blob.polarity)


    total_positive_words = len(positive_words_list)
    total_negative_words = len(negative_words_list)
    for i in positive_words_polarity:
        total_positive_polarity = total_positive_polarity + i
    for i in negative_words_polarity:
        total_negative_polarity = total_negative_polarity + i

    if total_count_without_punch != 0:
        p_ratio = (float(total_positive_words) / float(total_count_without_punch))
    if total_count_without_punch != 0:
        n_ratio = (float(total_negative_words) / float(total_count_without_punch))
    sentiment_score = p_ratio - n_ratio

    app_title = str(app_title)
    app_title = re.sub(r'\xfa', '', app_title)
    app_title = re.sub(r'\xa0', '', app_title)
    app_title = re.sub(r'\u2105', '', app_title)
    app_title = re.sub(r'\u2b50', '', app_title)
    app_title = re.sub(r'\u2755', '', app_title)
    app_title = re.sub(r'\u05d2', '', app_title)
    app_title = re.sub(r'\u20b9', '', app_title)

    app_description = str(app_description)
    app_description = re.sub(r'\xfa', '', app_description)
    app_description = re.sub(r'\xa0', '', app_description)
    app_description = re.sub(r'\u2105', '', app_description)
    app_description = re.sub(r'\u2b50', '', app_description)
    app_description = re.sub(r'\u2755', '', app_description)
    app_description = re.sub(r'\u05d2', '', app_description)
    app_description = re.sub(r'\u20b9', '', app_description)

    app_raw_title = app_title
    app_raw_title = text_to_lower(app_raw_title)
    app_title = text_to_lower(app_title)
    app_title = app_title.strip()
    app_title = remove_stop_words(app_title)
    app_title = remove_punctuation(app_title)
    app_title = remove_single_characters(app_title)
    app_title = app_title.strip()

    app_description = text_to_lower(app_description)
    app_description = app_description.strip()
    app_description = remove_stop_words(app_description)
    app_description = remove_punctuation(app_description)
    app_description = remove_single_characters(app_description)
    app_description = app_description.strip()
    app_raw_title_token = app_raw_title.split()

    app_title_token = app_title.split()
    app_description_token = app_description.split()

    cosin_app_title_and_review_text = textdistance.cosine(app_title_token, text_without_stopwords)
    cosin_app_summary_and_review_text = textdistance.cosine(app_description_token, text_without_stopwords)

    constructed_features= pd.DataFrame()
    constructed_features['review_score']=review_score
    constructed_features['total_count_with_punch'] = total_count_with_punch
    constructed_features['total_chars_count_with_punch'] = total_chars_count_with_punch
    constructed_features['total_count_without_punch'] = total_count_without_punch
    constructed_features['total_chars_count_without_punch'] = total_chars_count_without_punch
    constructed_features['unique_words_count'] = unique_words_count
    constructed_features['total_positive_words'] = round(total_positive_words, 4)
    constructed_features['total_negative_words'] = round(total_negative_words, 4)
    constructed_features['total_positive_polarity'] = round(total_positive_polarity, 4)
    constructed_features['total_negative_polarity'] = round(total_negative_polarity, 4)
    constructed_features['p_ratio'] = round(p_ratio, 4)
    constructed_features['n_ratio'] = round(n_ratio, 4)
    constructed_features['sentiment_score'] = round(sentiment_score, 4)
    constructed_features['cosin_app_title_and_review_text'] = round(cosin_app_title_and_review_text, 4)
    constructed_features['cosin_app_summary_and_review_text'] = round(cosin_app_summary_and_review_text, 4)


    # Final Features List
    # f[0] = review_score
    # f[1] = total_count_with_punch
    # f[2] = total_chars_count_with_punch
    # f[3] = total_count_without_punch
    # f[4] = total_chars_count_without_punch
    # f[5] = unique_words_count
    # f[6] = round(total_positive_words, 4)
    # f[7] = round(total_negative_words, 4)
    # f[8] = round(total_positive_polarity, 4)
    # f[9] = round(total_negative_polarity, 4)
    # f[10] = round(p_ratio, 4)
    # f[11] = round(n_ratio, 4)
    # f[12] = round(sentiment_score, 4)
    #
    # f[13] = round(cosin_app_title_and_review_text, 4)
    # f[14] = round(cosin_app_summary_and_review_text, 4)
    features_data= {'review_score':[review_score], 'total_count_with_punch':[total_count_with_punch],
                    'total_chars_count_with_punch':[total_chars_count_with_punch],'total_count_without_punch':[total_count_without_punch],
                    'total_chars_count_without_punch':[total_chars_count_without_punch],'unique_words_count':[unique_words_count],
                    'total_positive_words':[round(total_positive_words, 4)],'total_negative_words':[round(total_negative_words, 4)],
                    'total_positive_polarity':[round(total_positive_polarity, 4)],'total_negative_polarity':[round(total_negative_polarity, 4)],
                    'p_ratio':[round(p_ratio, 4)],'n_ratio':[round(n_ratio, 4)],
                    'sentiment_score':[round(sentiment_score, 4)],'cosin_app_title_and_review_text':[round(cosin_app_title_and_review_text, 4)],
                    'cosin_app_summary_and_review_text':[round(cosin_app_summary_and_review_text, 4)]}
    df=pd.DataFrame(features_data, columns=["review_score","total_count_with_punch","total_chars_count_with_punch"
         ,"total_count_without_punch","total_chars_count_without_punch","unique_words_count"
         ,"total_positive_words","total_negative_words","total_positive_polarity"
         ,"total_negative_polarity","p_ratio","n_ratio","sentiment_score"
         ,"cosin_app_title_and_review_text","cosin_app_summary_and_review_text"])


    return df
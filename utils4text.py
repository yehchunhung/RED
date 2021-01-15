"""
utils4text.py is the script file storing many useful functions for processing the comment dataframes from the subreddits.
That is, it is mainly used for text EDA.

Made by Chun-Hung Yeh.
"""

import numpy as np
import pandas as pd
import multiprocess as mp
import re
import nltk
import contractions
import string
from emoji import UNICODE_EMOJI
from itertools import repeat
from collections import Counter
from nltk import pos_tag, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from joblib import Parallel, delayed
from profanity_check import predict_prob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def build_convs(df):
    """
    Use parallel computing. Consider only one post at each time.
    Reconstruct the dataframe to a more conversation-like dataframe.

    Arg:
        df: A given dataframe scraped from a certain subreddit.
    Return:
        df_convs: A more conversation-like dataframe with the columns such as 
        conversation ID, subreddit, post title, author, dialog turn, and text.
    """
    # initialize conversation dataframe
    df_convs = pd.DataFrame(columns = ['subreddit', 'post title', 'author', 'dialog turn', 'text'])
    
    # consider each post
    df_link_id = df.reset_index().drop('index', axis = 1)
    row_list = []
    convs_turn = 0

    # add post from df_link_id
    post_row = df_link_id.loc[0, :]
    convs_turn += 1
    row_list.append({'subreddit': post_row['subreddit'], 'post title': post_row['title'], 
                     'author': post_row['post_author'], 'dialog turn': convs_turn, 'text': post_row['post_content']})

    # iterate over each comment from df_link_id
    for i, row in df_link_id.iterrows():
        convs_turn += 1
        row_list.append({'subreddit': row['subreddit'], 'post title': row['title'],
                         'author': row['comment_author'], 'dialog turn': convs_turn, 'text': row['comment_content']})

    df_convs = df_convs.append(pd.DataFrame(row_list))
    
    # change data types
    df_convs['dialog turn'] = df_convs['dialog turn'].astype('int32')

    return df_convs


def apply_parallel(grouped_df, func):
    """
    Parallelize the 'build_convs' function by grouping each post and its comments.
    And then concatenate all of them into a complete dataframe.

    Arg:
        grouped_df: A dataframe on which groupby function is applied.
    Return:
        pd.concat(retLst): A complete dataframe with the conversation sets between posts and comments.  
    """
    retLst = Parallel(n_jobs = mp.cpu_count())(delayed(func)(group) for id, group in grouped_df)
    return pd.concat(retLst)


def build_concise_convs_df(df_convs, njobs = mp.cpu_count()):
    """
    Using the functions, build_convs and apply_parallel, a dataframe with conversation sets
    can be easily built. Also the id for each conversation is added.

    Arg:
        df_convs: The original dataframe consisting of posts and comments parsed from the text files.
    Return:
        df_convs_concise: The concise version of a dataframe with conversation sets.
    """
    df_convs_concise = apply_parallel(df_convs.groupby(df_convs.link_id), build_convs)
    df_convs_concise['conversation id'] = (df_convs_concise.groupby(['post title']).cumcount() == 0).astype(int)
    df_convs_concise['conversation id'] = df_convs_concise['conversation id'].cumsum()
    df_convs_concise = df_convs_concise[['conversation id', 'subreddit', 'post title', 'author', 'dialog turn', 'text']]
    df_convs_concise = df_convs_concise.reset_index().drop('index', axis = 1)
    return df_convs_concise


def remove_marks(text):
    """
    Remove those unnecessary marks inside texts.

    Arg:
        text: A string that could be either posts or comments.
    Return:
        new_text: A string which is a clean sentence.
    """
    # remove HTML tags 
    new_text = re.sub('<.*?>', '', text) 

    # remove URL
    new_text = re.sub('http\S+', '', new_text)  

    # replace number with <NUM> token
    new_text = re.sub('\d+', ' NUM ', new_text)

    return new_text


def get_wordnet_pos(tag):
    """
    Transform a positional tag to its corresponding WordNet format.

    Arg:
        tag: A positional tag from pos_tag function.
    Return:
        The associated wordnet format given by a tag.
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def token_lemmatize(token, lemmatizer):
    """
    Lemmatize a token to convert a token back to its root form.
    When dealing with punctuation marks or emojis, simply return them as usual.

    Arg:
        token: A word in the string type.
        lemmatizer: The object from wordnet lemmatizer.
    Return:
        token in its root form.
    """
    if token == 'NUM':
        # keep NUM token as usual
        return token
    elif token in string.punctuation:
        # keep punctuation marks as usual
        return token
    elif token in UNICODE_EMOJI:
        # keep emojis
        return token
    elif token.isalpha(): 
        # consider English words
        token, tag = pos_tag([token])[0][0], pos_tag([token])[0][1]
        return lemmatizer.lemmatize(token, get_wordnet_pos(tag))
    # else:
    #     # transform those nonwords as the token NOWORD
    #     token = 'NONWORD'
    #     return token
    

def text_lemmatize(text, lemmatizer):
    """
    Apply lemmatization on the raw texts to convert the words in texts back to their root
    forms. Before lemmatization, remove unnecessary marks and stopwords to keep only the 
    meaningful words.

    Arg:
        text: A string text.
        lemmatizer: An object of WordNetLemmatizer.
    Return:
        lem_words: A list of lemmatized words.
    """
    # remove unnecessary marks and tokenize
    tokens = word_tokenize(remove_marks(text))

    # remove stopwords
    filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]

    # lemmatize the tokenized texts
    lem_words = []
    lem_words += list(map(token_lemmatize, filtered_tokens, repeat(lemmatizer)))
        
    return lem_words


def compute_tokens(subreddit_convs_concise):
    """
    Given the text data from a subreddit, lemmatize and compute the word tokens using the defined function, text_lemmatize.
    Before that, remove the newline tag and expanding the English contraction.
    The reason why the progress_bar is set to false is because of Google Colab's memory limitation.
    If it's not the problem in your local machine, you could simply convert it to be true to check the processing status.
    
    Arg:
        subreddit_convs_concise: A conversation dataframe from a subreddit.
    Return:
        subreddit_tokens: A series with each row containing a list of word tokens from either post or comment.
    """
    # copy the text column from original dataframe
    subreddit_text = subreddit_convs_concise['text'].copy()

    # expanding contraction
    subreddit_text = subreddit_text.swifter.progress_bar(False).apply(lambda text: text.replace('\n', ' '))\
                                   .swifter.progress_bar(False).apply(lambda text: ' '.join([contractions.fix(word) for word in text.split()]))

    # lemmatize
    lemmatizer = WordNetLemmatizer()
    subreddit_tokens = subreddit_text.swifter.progress_bar(False).apply(lambda text: text_lemmatize(text, lemmatizer))

    return subreddit_tokens


def compute_turn_distribution(df):
    """
    Given a conversation dataframe from a subreddit (note that the dataframe is in the concise format indicated by Supervisor),
    find out the dialog turn distribution.

    Arg:
        df: A conversation dataframe from a subreddit.
    Return:
        turn_dist: A series about dialog turn distribution.
    """
    turn_dist = df.groupby('conversation id').size().value_counts().sort_index()
    turn_dist = pd.DataFrame(turn_dist).reset_index().rename(columns = {'index': 'turns', 0: 'count'})

    return turn_dist


def extract_turn_10_more(df):
    """
    Given a concise conversation dataframe, extract those with 10 or more dialog turns.

    Arg:
        df: A conversation dataframe from a subreddit.
    Return:
        turn_10_more: A dataframe containing only those conversations with 10 or more turns.
    """
    turn_dist = df.groupby('conversation id').size()
    turn_dist_10_more_index = turn_dist[turn_dist >= 10].index
    turn_10_more = df[df['conversation id'].isin(list(turn_dist_10_more_index))]
    
    return turn_10_more


def remove_newline(df):
    """
    For each text in either post or comment, remove the newline tag.
    
    Arg:
        df: A given conversation dataframe from a certain subreddit.
    Return:
        df: A cleaner conversation dataframe without the newline tags.
    """
    df['text'] = df['text'].swifter.progress_bar(False).apply(lambda text: text.replace('\n', ' '))
    df['text'] = df['text'].swifter.progress_bar(False).apply(lambda text: text.replace("\\", ''))
    return df


def remove_toxicity(df):
    """
    Use parallel computing. Consider only one post at each time.
    In each post, detect the toxicity and remove the following dialog turns.

    Arg:
        df: A given conversation dataframe from a certain subreddit.
    Return:
        df_clean: A cleaner version of the conversation dataframe with no toxic words.
    """
    # initialize clean conversation dataframe
    df_clean = pd.DataFrame(columns = ['conversation id', 'subreddit', 'post title', 'author', 'dialog turn', 'text'])
    
    # consider each post
    df_post = df.reset_index().drop('index', axis = 1)
    clean_row_list = []

    # iterate over each comment from df_link_id
    for i, row in df_post.iterrows():
        if predict_prob([row['text']])[0] > 0.95 and row['dialog turn'] > 1:
            break
        else:
            clean_row_list.append({'conversation id': row['conversation id'], 'subreddit': row['subreddit'],
                                   'post title': row['post title'], 'author': row['author'],
                                   'dialog turn': row['dialog turn'], 'text': row['text']})
        
    df_clean = df_clean.append(pd.DataFrame(clean_row_list))

    return df_clean


def extract_toxicity(df):
    """
    Use parallel computing. Consider only one post at each time.
    In each post, extract the toxic texts.

    Arg:
        df: A given conversation dataframe from a certain subreddit.
    Return:
        df_toxic: A conversation dataframe with exclusively toxic words.
    """
    # initialize clean conversation dataframe
    df_toxic = pd.DataFrame(columns = ['conversation id', 'subreddit', 'post title', 'author', 'dialog turn', 'text'])

    # consider each post
    df_post = df.reset_index().drop('index', axis = 1)
    toxic_row_list = []

    # iterate over each comment from df_link_id
    for i, row in df_post.iterrows():
        if predict_prob([row['text']])[0] > 0.95 and row['dialog turn'] > 1:
            # record the toxic text
            toxic_row_list.append({'conversation id': row['conversation id'], 'subreddit': row['subreddit'],
                                   'post title': row['post title'], 'author': row['author'],
                                   'dialog turn': row['dialog turn'], 'text': row['text']})
    
    df_toxic = df_toxic.append(pd.DataFrame(toxic_row_list))

    return df_toxic


def differentiate_clean_toxic_convs_df(df_convs, njobs = mp.cpu_count()):
    """
    Applying the profanity checking functions, differentiate clean conversations and toxic ones from a raw conversation dataframe.
    
    Arg:
        df_convs: A given conversation dataframe from a certain subreddit.
    Return:
        df_clean: A cleaner conversation dataframe without profanity.
        df_toxic: A toxic conversation dataframe with only profanity.
    """
    df_clean = apply_parallel(df_convs.groupby(df_convs['conversation id']), remove_toxicity)
    df_toxic = apply_parallel(df_convs.groupby(df_convs['conversation id']), extract_toxicity)
    df_clean = df_clean.reset_index().drop('index', axis = 1)
    df_toxic = df_toxic.reset_index().drop('index', axis = 1)

    return (df_clean, df_toxic)


def find_speaker_frequent_words(df, num):
    """
    Given a conversation dataframe from a certain subreddit, find the top frequent words spoken by speakers.

    Args:
        df: A specified dataframe from a subreddit.
        num: A ranking number used for finding the top frequent words.
             For example, if num = 5, then we'll find the top 5 frequent words.
    Return:
        result: A dataframe showing the top frequent words. 
    """
    # extract speakers' turn
    df_speaker = df[df['dialog turn'] == 1]

    # compute tokens
    df_speaker_filtered = compute_tokens(df_speaker)

    # find top (num) frequent words
    result = pd.DataFrame(Counter(df_speaker_filtered.sum()).most_common(num), columns = ["word", "count"])

    return result


def find_listener_frequent_words(df, num):
    """
    Given a conversation dataframe from a certain subreddit, find the top frequent words spoken by listeners.

    Args:
        df: A specified dataframe from a subreddit.
        num: A ranking number used for finding the top frequent words.
    Return:
        result: A dataframe showing the top frequent words. 
    """
    # extract listeners' turn
    df_listener = df[df['dialog turn'] != 1]

    # compute tokens
    df_listener_filtered = compute_tokens(df_listener)

    # find top (num) frequent words
    result = pd.DataFrame(Counter(df_listener_filtered.sum()).most_common(num), columns = ["word", "count"])
    return result


def extract_profanity(df):
    """
    Use parallel computing. Consider only one post at each time.
    In each post, extract the profanity considering post texts and comments. 

    Arg:
        df: A given conversation dataframe from a certain subreddit.
    Return:
        df_toxic: A conversation dataframe with exclusively toxic words.
    """
    # initialize clean conversation dataframe
    df_toxic = pd.DataFrame(columns = ['conversation id', 'subreddit', 'post title', 'author', 'dialog turn', 'text'])

    # consider each post
    df_post = df.reset_index().drop('index', axis = 1)
    toxic_row_list = []

    # iterate over each comment from df_link_id
    for i, row in df_post.iterrows():
        if predict_prob([row['text']])[0] > 0.95:
            # record the toxic text
            toxic_row_list.append({'conversation id': row['conversation id'], 'subreddit': row['subreddit'],
                                   'post title': row['post title'], 'author': row['author'],
                                   'dialog turn': row['dialog turn'], 'text': row['text']})
    
    df_toxic = df_toxic.append(pd.DataFrame(toxic_row_list))

    return df_toxic


def count_speaker_profanity(df):
    """
    Compute the number of profane words spoken by the speakers.

    Arg:
        df: A dataframe containing profane utterances.
    Return:
        len(df_speaker_toxic): the length of the dataframe of profanity from speakers.
    """
    # extract speakers' turn
    df_speaker = df[df['dialog turn'] == 1]

    # extract toxic turn
    df_speaker_toxic = apply_parallel(df_speaker.groupby(df_speaker['conversation id']), extract_profanity)

    return len(df_speaker_toxic)


def count_listener_profanity(df):
    """
    Compute the number of profane words spoken by the listeners.

    Arg:
        df: A dataframe containing profane utterances.
    Return:
        len(df_listener_toxic): the length of the dataframe of profanity from listeners.
    """
    # extract listeners' turn
    df_listener = df[df['dialog turn'] != 1]

    # extract toxic turn
    df_listener_toxic = apply_parallel(df_listener.groupby(df_listener['conversation id']), extract_profanity)

    return len(df_listener_toxic)


def lang_correct(subreddit_convs_clean, tool):
    """
    Fix the utterances to the correct English format.

    Arg:
        subreddit_convs_clean: A dataframe containing clean conversations.
        tool: A toolkit for language correction.
    Return:
        subreddit_convs_clean: A dataframe with clean and grammatically correct conversations.
    """
    # language correct
    subreddit_convs_clean['text'] = subreddit_convs_clean['text'].swifter.apply(lambda text: tool.correct(text))

    return subreddit_convs_clean


def determine_pos_neu_neg(compound):
    """
    Based on the compound score, classify a sentiment into positive, negative, or neutral.

    Arg:
        compound: A numerical compound score.
    Return:
        A label in "positive", "negative", or "neutral".
    """
    if compound >= 0.05:
        return 'positive'
    elif compound < 0.05 and compound > -0.05:
        return 'neutral'
    else:
        return 'negative'

def compute_sentiment(subreddit_convs_clean, analyzer = SentimentIntensityAnalyzer()):
    """
    Calculate the conversation sentiment via the built sentiment analyzer.

    Args:
        subreddit_convs_clean: A dataframe containing clean conversations.
        analyzer: A built analyzer for sentiment analysis.
    Return:
        subreddit_convs_clean: A dataframe added with sentiment prediction.
    """
    # transform text column to string type
    subreddit_convs_clean['text'] = subreddit_convs_clean['text'].astype(str)

    # add one column to specify the value of sentiment
    subreddit_convs_clean['compound'] = subreddit_convs_clean['text']\
                                            .swifter.apply(lambda text: analyzer.polarity_scores(text)['compound'])
    
    # add another column for determining the type of sentiment
    subreddit_convs_clean['sentiment'] = subreddit_convs_clean['compound']\
                                            .swifter.apply(determine_pos_neu_neg)

    return subreddit_convs_clean


def speaker_sentiment_count(subreddit_convs):
    subreddit_speaker_convs = subreddit_convs[subreddit_convs['dialog turn'] == 1]
    return subreddit_speaker_convs['sentiment'].value_counts()


def listener_sentiment_count(subreddit_convs):
    subreddit_listener_convs = subreddit_convs[subreddit_convs['dialog turn'] != 1]
    return subreddit_listener_convs['sentiment'].value_counts()


def remove_nonnumeric_id(df):
    df['conversation id'] = df['conversation id'].astype(str)
    return df[df['conversation id'].apply(lambda x: x.isnumeric())]


def speaker_emotion_count(df_convs):
    """
    Compute the number of emotion prediction.

    Arg:
        df_convs: A conversation dataframe with predicted emotion labels.
    Return:
        df_speaker_emotion_count: A dataframe summarizing predicted emotion counting. 
    """
    if len(df_convs['dialog turn']) >= 1:
        speaker = df_convs[df_convs['dialog turn'] == 1].author.values[0]
        df_convs_speaker = df_convs[df_convs['author'] == speaker]
        df_speaker_emotion_count = pd.DataFrame(df_convs_speaker['emotion prediction'].value_counts())\
                                        .reset_index()\
                                        .rename(columns = {'index': 'emotion', 'emotion prediction': 'count'})
        return df_speaker_emotion_count
    else:
        return None


def listener_emotion_count(df_convs):
    """
    Compute the number of emotion prediction.

    Arg:
        df_convs: A conversation dataframe with predicted emotion labels.
    Return:
        df_listener_emotion_count: A dataframe summarizing predicted emotion counting. 
    """
    if len(df_convs['dialog turn']) >= 1:
        speaker = df_convs[df_convs['dialog turn'] == 1].author.values[0]
        df_convs_listener = df_convs[df_convs['author'] != speaker]
        df_listener_emotion_count = pd.DataFrame(df_convs_listener['emotion prediction'].value_counts())\
                                        .reset_index()\
                                        .rename(columns = {'index': 'emotion', 'emotion prediction': 'count'})

        return df_listener_emotion_count
    else:
        return None
        

def apply_parallel_2(grouped_df, func):
    retLst = Parallel(n_jobs = mp.cpu_count())(delayed(func)(group) for id, group in grouped_df if len(grouped_df) >= 1)
    return pd.concat(retLst)


def build_speaker_emotion_count(df_convs):
    total_speaker_emotion_count = apply_parallel_2(df_convs.groupby(df_convs['conversation id']), speaker_emotion_count)
    total_speaker_emotion_count = total_speaker_emotion_count.groupby('emotion')['count'].count()
    return total_speaker_emotion_count


def build_listener_emotion_count(df_convs):
    total_listener_emotion_count = apply_parallel_2(df_convs.groupby(df_convs['conversation id']), listener_emotion_count)
    total_listener_emotion_count = total_listener_emotion_count.groupby('emotion')['count'].count()
    return total_listener_emotion_count
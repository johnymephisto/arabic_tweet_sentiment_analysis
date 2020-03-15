import base64

import keras.backend.tensorflow_backend as tb
import pandas as pd
import streamlit as st

tb._SYMBOLIC_SCOPE.value = True

from classifier import Classifier
from tweetmanger import TweetManager
from results import Results


# Load classification model
@st.cache(allow_output_mutation=True)
def load_model():
    with st.spinner('Loading classification model...'):
        classifier = Classifier()

    return classifier


# Load twitter API endpoint
@st.cache(allow_output_mutation=True)
def init_twitter():
    with st.spinner('Loading Twitter Manager...'):
        tweet_manager = TweetManager()

    return tweet_manager


# Load twitter API endpoint
@st.cache(allow_output_mutation=True)
def get_twitter_data(tweet_manager, tweet_input, sidebar_result_type, sidebar_tweet_count):
    df = tweet_manager.get_tweets(tweet_input, result_type=sidebar_result_type, count=sidebar_tweet_count)

    return df


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download as csv file (Right click and save link as csv)</a>'
    return href


def main():
    st.title("Sentiment Analyzer on Twitter Hashtags in Arabic Language")

    st.sidebar.header("Tweet Parsing options")

    sidebar_tweet_count = st.sidebar.slider(label='Number of tweets',
                                            min_value=10,
                                            max_value=1000,
                                            value=10,
                                            step=10)

    sidebar_result_type = st.sidebar.selectbox('Result Type', ('popular', 'mixed', 'recent'), index=0)

    pd.set_option('display.max_colwidth', 0)

    classifier = load_model()
    tweet_manager = init_twitter()
    results = Results()

    st.subheader('Input the hashtag to analyze')

    tweet_input = st.text_input('Hashtag:')
    print(f'Getting tweets for {tweet_input}')

    if tweet_input != '':
        # Get tweets
        with st.spinner('Parsing from twitter API'):
            df = get_twitter_data(tweet_manager, tweet_input, sidebar_result_type, sidebar_tweet_count)

        # st.dataframe(df)
        # Make predictions
        if df.__len__() > 0:
            with st.spinner('Predicting...'):
                pred = classifier.get_sentiment(df)

                st.subheader('Prediction:')
                st.dataframe(pred)

                st.markdown(get_table_download_link(df), unsafe_allow_html=True)

            results.calculate_results(df)

            with st.spinner('Generating Visualizations...'):
                st.header('Visualizations')
                st.subheader('Pie Chart')
                st.plotly_chart(results.get_pie_chart_counts())
                st.subheader('Bar Chart')
                st.plotly_chart(results.get_bar_chart_counts())
                st.subheader('Pie Chart showing most used words')
                st.plotly_chart(results.get_pie_chart_most_counts())
                st.subheader('Bar Chart showing most used words')
                st.plotly_chart(results.get_bar_chart_most_counts())

                st.subheader('Time series showing tweets')
                st.plotly_chart(results.get_line_chart_tweets())

                st.header('Tables')
                st.subheader('Hashtag Analysis')
                st.write('Number of tweets per classification')
                st.table(results.get_stats_table())

                st.subheader('Most words frequency')
                st.write('Top 15 words from tweets')
                st.table(results.get_stats_table_most_counts())


        else:
            st.write('No tweets found')


if __name__ == '__main__':
    main()

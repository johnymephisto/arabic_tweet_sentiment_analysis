import pickle

from keras.preprocessing.sequence import pad_sequences
# from keras.models import load_model
from tensorflow.keras.models import load_model


class Classifier(object):

    def __init__(self):
        """
        The models are preloaded so that it wont take time during inference
        """
        self.model = load_model('SentimentAnalysisArabic/models/arabic_sentiment_lstm.hdf5')
        with  open("SentimentAnalysisArabic/models/arabic_sentiment_lstm.pickle", "rb") as f:
            self.tokenizer = pickle.load(f)

    def get_sentiment(self, df):
        """
        Takes a text input that you want to run sentiment analysis on.
        Returns with sentiment score and sentiment class (positive or negative)

        :param text_input: Text to run sentiment analysis on
        :return: (sentiment_score, sentiment_class)
        """
        print(df)
        sequences = self.tokenizer.texts_to_sequences(df['tweet'])
        data = pad_sequences(sequences, maxlen=100)
        num_class = self.model.predict(data)
        df['sentiment_score'] = num_class

        def score_segregate(value):
            if value <= 0.35:
                return 'Negative'
            elif value > 0.35 and value < 0.65:
                return 'Neutral'
            elif value >= 0.65:
                return 'Positive'

        df['sentiment_class'] = df['sentiment_score'].apply(score_segregate)

        return df


def main():
    """
    To test the classifier
    """
    print(Classifier().get_sentiment(
        'يواصل #فيروس_كورونا الحاق المزيد من الاضرار بقطاع النقل الجوي، حيث صرح العضو المنتدب لشركة انه سيتم '))


if __name__ == '__main__':
    main()

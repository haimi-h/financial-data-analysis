import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def load_data(file_path):
    return pd.read_csv(file_path)

def analyze_headline_length(df, index):
    headline = df.loc[index, 'headline']
    return len(headline)

def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def analyze_topics(df, sample_size=100, num_topics=5):
    sample_df = df.sample(n=sample_size, random_state=1)
    sample_df['clean_text'] = sample_df['headline'].apply(preprocess_text)
    dictionary = corpora.Dictionary(sample_df['clean_text'])
    corpus = [dictionary.doc2bow(text) for text in sample_df['clean_text']]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    return lda_model

def plot_publication_frequency_over_time(df):
    df['date'] = pd.to_datetime(df['date'], utc=True)
    publication_frequency = df.groupby(df['date'].dt.date).size()
    plt.figure(figsize=(10, 6))
    plt.plot(publication_frequency.index, publication_frequency.values, marker='o', linestyle='-')
    plt.title('Publication Frequency Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Publications')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_distribution_of_publication_times(df):
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['hour'] = df['date'].dt.hour
    publication_times = df['hour'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    plt.bar(publication_times.index, publication_times.values, color='skyblue')
    plt.title('Distribution of Publication Times')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Publications')
    plt.xticks(range(24))
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def plot_distribution_of_articles_by_publisher(df, sample_size=None):
    if sample_size:
        sample_df = df.sample(n=sample_size, random_state=42)
        publisher_counts = sample_df['publisher'].value_counts()
        title = f'Distribution of News Articles by Publisher (Sample Size: {sample_size})'
    else:
        publisher_counts = df['publisher'].value_counts()
        title = 'Distribution of News Articles by Publisher'
    plt.figure(figsize=(10, 6))
    publisher_counts.plot(kind='bar', color='skyblue')
    plt.title(title)
    plt.xlabel('Publisher')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_distribution_of_articles_by_domain(df):
    df['domain'] = df['publisher'].str.split('@').str[1]
    domain_counts = df['domain'].value_counts().head(10)
    domain_counts.plot(kind='bar', figsize=(10, 6), color='skyblue')
    plt.title('Distribution of News Articles by Domain')
    plt.xlabel('Domain')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def main():
    file_path = 'raw_analyst_ratings.csv'
    df = load_data(file_path)

    # Analyze headline length
    headline_length = analyze_headline_length(df, index=0)
    print("Length of the specific headline:", headline_length)

    # Analyze sentiment
    sample_df = df.head(20)  # Selecting the first rows as an example
    sample_df['sentiment'] = sample_df['headline'].apply(analyze_sentiment)
    print(sample_df[['headline', 'sentiment']])

    # Analyze topics
    lda_model = analyze_topics(df, sample_size=100, num_topics=5)
    for idx, topic in lda_model.print_topics():
        print(f'Topic {idx}: {topic}')

    # Plot publication frequency over time
    plot_publication_frequency_over_time(df)

    # Plot distribution of publication times
    plot_distribution_of_publication_times(df)

    # Plot distribution of articles by publisher
    plot_distribution_of_articles_by_publisher(df, sample_size=100)

    # Plot distribution of articles by domain
    plot_distribution_of_articles_by_domain(df)

if __name__ == "__main__":
    main()

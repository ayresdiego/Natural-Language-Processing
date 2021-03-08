

import pandas as pd
import matplotlib.pyplot as plt


def df_adjust(df, y_col, x_col):
    print(f"\n{'__' * 50}\nAdjust the DF")

    print(df.columns)
    print(df.head())
    print(f"\nAmount of rows: {len(df.index)}")

    print(f"\nAmount of Null Values\n{df.isnull().sum()}\n")
    df.dropna(subset=[x_col]
              , inplace=True)


    blank_list = []
    df_temp = df.loc[:,[y_col, x_col]].copy()
    for index, y, x in df_temp.itertuples():
        if type(x) == str:
            if x.isspace():
                print(f"True Blank Space for index: {index}")
                blank_list.append(index)

    print(f"\nAmount of Blank Message: {len(blank_list)}")
    df.drop(blank_list, inplace=True)


    print(f"\nUnique labels: {df.loc[:, y_col].unique()}")
    print(f"\nAmounts per unique labels: \n{df.loc[:, y_col].value_counts()}") # amount of rows per unique value in y_column


    df["nb_chars"] = df["review"].apply(lambda x: len(x))

    df["nb_words"] = df["review"].apply(lambda x: len(x.split(" ")))
    print(df.head().to_string())

    return df


def clean_text(text):
    from nltk.corpus import wordnet

    def get_wordnet_pos(pos_tag):
        if pos_tag.startswith('J'):
            return wordnet.ADJ
        elif pos_tag.startswith('V'):
            return wordnet.VERB
        elif pos_tag.startswith('N'):
            return wordnet.NOUN
        elif pos_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    import string
    from nltk import pos_tag
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer


    text = text.lower()
    text = [word.strip(string.punctuation) for word in text.split(" ")]

    text = [word for word in text if not any(c.isdigit() for c in word)]

    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]

    text = [t for t in text if len(t) > 0]

    pos_tags = pos_tag(text)

    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]

    text = [t for t in text if len(t) > 1]

    text = " ".join(text)
    return (text)


def show_wordcloud(data, title=None, background_color_x="white"):
    from wordcloud import WordCloud
    wordcloud = WordCloud(
        background_color=background_color_x,
        max_words=200,
        max_font_size=40,
        scale=3,
        random_state=42
    ).generate(str(data))

    fig = plt.figure(1, figsize=(20, 20))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


def text_sentimental_analysis(df):
    print(f"\n{'__'*50}\nSentimental Analysis")

    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    a = 'This was a good movie.'
    score = sid.polarity_scores(a)
    print(score)


    df['scores_dict'] = df['review'].apply(lambda x: sid.polarity_scores(x))
    df = pd.concat(
        [df, df['scores_dict'].apply(pd.Series)]
        , axis=1)
    df = df.drop(['scores_dict'], axis=1)


    compound_limit = 0.3
    df['y_prediction'] = df['compound'].apply(lambda c: 'pos' if c >=compound_limit else 'neg')
    print(df.columns)
    df.to_csv("Sentimental_prediction.csv")
    print(df.head().to_string())


    print(f"\n{'__'*50}\nEvaluation of sentimental Analysis. Not always possible")
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    print(confusion_matrix(df['label'], df['y_prediction']))

    print(classification_report(df['label'], df['y_prediction']))

    accuracy = accuracy_score(df['label'], df['y_prediction'])
    print("\nAccuracy:\033[1;34;40m{:.1%}\033[m".format(accuracy))

    print('Weighted F1-score: \033[1;34;40m{:.1%}\033[m'.format(f1_score(df['label'], df['y_prediction'], average='weighted')))


    return df


def TfidfVectorizer_columns(df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(
        max_df=0.9
        , min_df=10
        , stop_words='english'
    )

    tfidf_result = tfidf.fit_transform(df[x_col]).toarray()

    tfidf_df = pd.DataFrame(tfidf_result, columns=tfidf.get_feature_names())
    tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
    tfidf_df.index = df.index
    tfidf_df = pd.concat([df, tfidf_df], axis=1)
    print(tfidf_df.head().to_string())


    return tfidf_df


def modeling_most_important_features_for_prediction(df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    df = df.rename(columns=lambda x: x[5:] if x.startswith('word_') else x)
    print(df.columns)

    label = "y_prediction"
    ignore_cols = ["label", "review", "review",  "nb_chars", "nb_words", "neg", "neu", "pos", "compound", "y_prediction"]
    features = [c for c in df.columns if c not in ignore_cols]


    X_train, X_test, y_train, y_test = train_test_split(df[features], df[label], test_size = 0.20, random_state = 42)


    rf = RandomForestClassifier(n_estimators = 100, random_state = 42)
    rf.fit(X_train, y_train)


    feature_importance_df = pd.DataFrame({"feature": features, "importance": rf.feature_importances_})
    feature_importance_df.sort_values("importance", ascending=False, inplace=True)
    print(feature_importance_df.head(20))

    show_wordcloud(feature_importance_df, background_color_x="black")

    return


import os
cwd = os.getcwd()
file = r'amazon_reviews.tsv'
file = r'movie_reviews.tsv'
df = pd.read_csv(os.path.join(cwd, file)
                 , sep='\t'
                 , nrows=500
                 )

y_col = 'label'
x_col = 'review'

df = df_adjust(df, y_col, x_col)

df[x_col] = df[x_col].apply(lambda x: clean_text(x))


show_wordcloud(df[x_col])

df = text_sentimental_analysis(df)

df = TfidfVectorizer_columns(df)

modeling_most_important_features_for_prediction(df)


import pandas as pd


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
    print(f"\nAmounts per unique labels: \n{df.loc[:, y_col].value_counts()}")  # amount of rows per unique value in y_column

    return df


def text_classification_model(df, y_col, x_col):
    print(f"\n{'__'*50}\nText Classification Model")
    print(f"\n\nDependent Variable: {y_col}\nRegressors: {x_col}\n")

    from sklearn.model_selection import train_test_split
    X = df[x_col]
    y = df[y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)

    from sklearn.svm import LinearSVC
    clf = LinearSVC()
    clf.fit(X_train_tfidf, y_train)

    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LinearSVC()),
    ])

    text_clf.fit(X_train, y_train)


    print(f"\n{'__'*50}\nEvaluation of Classifier")
    y_predicted = text_clf.predict(X_test)

    from sklearn import metrics
    print(metrics.confusion_matrix(y_test, y_predicted))

    print(metrics.classification_report(y_test, y_predicted))

    accuracy = metrics.accuracy_score(y_test, y_predicted)
    print("\nAccuracy:\033[1;34;40m{:.1%}\033[m".format(accuracy))

    from sklearn.metrics import f1_score
    print('Weighted F1-score: \033[1;34;40m{:.1%}\033[m'.format(f1_score(y_test, y_predicted, average='weighted')))

    print(f"\n{'__'*50}\nPredictions")
    text_x = "The movie was great and wonderful"
    print(text_clf.predict([text_x]))

    return


import os
cwd = os.getcwd()
file = r'movie_reviews.tsv'
df = pd.read_csv(os.path.join(cwd, file)
                 , sep='\t'
                 )

y_col = 'label'
x_col = 'review'

df = df_adjust(df, y_col, x_col)


text_classification_model(df, y_col, x_col)





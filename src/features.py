from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.sparse import hstack 

def build_features(df, tfidf=None, scaler=None, fit=True):
    if fit:
        tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1,2)
        )
        X_text = tfidf.fit_transform(df["full_text"])
    else:
        X_text = tfidf.transform(df["full_text"])
# add more features bitwise symbols
    def extra_features(text): 
        return [
            len(text),
            # text.count('=') + text.count('+') + text.count('-'),
            sum(text.count(k) for k in ['=','+','-', '*','|']),
            sum(text.count(k) for k in ['dp','graph','tree','dfs','bfs'])
        ]

    X_extra = np.array(df["full_text"].apply(extra_features).tolist())

    if fit:
        scaler = StandardScaler()
        X_extra = scaler.fit_transform(X_extra)
    else:
        X_extra = scaler.transform(X_extra)

    X = hstack([X_text, X_extra])
    return X, tfidf, scaler



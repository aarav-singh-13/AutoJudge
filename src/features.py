from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from scipy.sparse import hstack 

import re

def constraint_features(text):
    text = text.lower()

    features = 0
    power_constraints = len(re.findall(r"10\^\d+", text))
    features += power_constraints

    large_numbers = len([int(x) for x in re.findall(r"\b\d+\b", text) if int(x) >= 100000])
    features += large_numbers

    constraint_keywords = [
        "constraint", "constraints", "at most",
        "no more than", "less than", "greater than"
    ]
    keyword_count = sum(text.count(k) for k in constraint_keywords)
    features += keyword_count

    inequality_count = sum(text.count(sym) for sym in ["<", ">", "≤", "≥"])
    features += inequality_count

    return features


def build_features(df, tfidf=None, scaler=None, fit=True):
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    if fit:
        tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1,2)
        )
        X_text = tfidf.fit_transform(df["full_text"])
    else:
        X_text = tfidf.transform(df["full_text"])

    def extra_features(text): 
        return [
            sum(text.count(k) for k in ['+','-','|', 'mod ', 'gcd', 'lcm']),
            sum(text.count(k) for k in ['subsequence', 'graph','tree']),
            constraint_features(text)
        ]

    X_extra = np.array(df["full_text"].apply(extra_features).tolist())

    if fit:
        scaler = StandardScaler()
        X_extra = scaler.fit_transform(X_extra)
    else:
        X_extra = scaler.transform(X_extra)

    X = hstack([X_text, X_extra])
    return X, tfidf, scaler



from sklearn.model_selection import train_test_split
from preprocess import preprocess_text
from features import build_features

df = preprocess_text("data/problems.json")


y_class = df["problem_class"]
y_score = df["problem_score"] 

df_train, df_test, y_class_train, y_class_test = train_test_split(
    df,
    y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)

X_train, tfidf, scaler = build_features(df_train, fit=True)
X_test, _, _ = build_features(df_test, tfidf=tfidf, scaler=scaler, fit=False)

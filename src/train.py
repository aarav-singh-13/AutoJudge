from sklearn.model_selection import train_test_split
from preprocess import preprocess_text
from features import build_features

df = preprocess_text("data/problems.jsonl")


y_class = df["problem_class"]
y_score = df["problem_score"] 

df_train, df_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
    df,
    y_class,
    y_score,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)


X_train, tfidf, scaler = build_features(df_train, fit=True)
X_test, _, _ = build_features(df_test, tfidf=tfidf, scaler=scaler, fit=False)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# clf = LogisticRegression(
#     max_iter=1000,
#     class_weight="balanced" 
# )

from sklearn.svm import LinearSVC

clf = LinearSVC(class_weight="balanced")
clf.fit(X_train, y_class_train)


clf.fit(X_train, y_class_train)

y_pred_class = clf.predict(X_test)
print(classification_report(y_class_test, y_pred_class))

joblib.dump(clf, "classifier.pkl")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

reg = LinearRegression()

reg.fit(X_train, y_score_train)
y_pred_score = reg.predict(X_test)
print("MSE:", mean_squared_error(y_score_test, y_pred_score))


joblib.dump(reg, "regressor.pkl")

joblib.dump(tfidf, "tfidf.pkl")
joblib.dump(scaler, "scaler.pkl")


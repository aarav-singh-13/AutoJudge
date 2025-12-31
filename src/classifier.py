from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from train import X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test

# clf = LogisticRegression(
#     max_iter=1000,
#     class_weight="balanced" 
# )

from sklearn.svm import LinearSVC

clf = LinearSVC(class_weight="balanced", max_iter=5000)
clf.fit(X_train, y_class_train)


clf.fit(X_train, y_class_train)

y_pred_class = clf.predict(X_test)
print(classification_report(y_class_test, y_pred_class))

joblib.dump(clf, "classifier.pkl")




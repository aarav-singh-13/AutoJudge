
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import joblib
from train import X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

clf = LinearSVC(class_weight="balanced", max_iter=5000)
clf.fit(X_train, y_class_train)


clf.fit(X_train, y_class_train)

y_pred_class = clf.predict(X_test)
print(classification_report(y_class_test, y_pred_class))

cm = confusion_matrix(y_class_test, y_pred_class)
fig, ax = plt.subplots(figsize=(8, 6))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['easy', 'hard', 'medium'])
disp.plot(cmap='Blues', ax=ax)
plt.title("Confusion Matrix")
plt.show()

joblib.dump(clf, "classifier.pkl")



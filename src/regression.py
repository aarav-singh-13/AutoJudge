from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
from train import X_train, X_test, y_score_train, y_score_test, tfidf, scaler

reg = LinearRegression()

reg.fit(X_train, y_score_train)
y_pred_score = reg.predict(X_test)
y_pred_score=np.clip(y_pred_score,0,10)
print("MSE:", mean_squared_error(y_score_test, y_pred_score))


joblib.dump(reg, "regressor.pkl")
joblib.dump(tfidf, "tfidf.pkl")
joblib.dump(scaler, "scaler.pkl")
# AutoJudge
Predicting Programming Problem Difficulty

## All the external libraries are present in requirements.txt

Dataset used : https://github.com/AREEG94FAHAD/TaskComplexityEval-24
(80% to train, 20% to test)

Approach and Models used :

1) Pre-processing : 
2) Feature Extraction : 
    used TF-IDF to monitor keywords
    hard-coded features like algebraic symbols and bitwise operators, algorithms, etc
3) Models :
    LinearSVC 
    Linear Regression

Evaluation Metrics :
    Accuracy : 48%
    MSE : 18.50214402554875
    RMSE : 4.3

Steps to run locally :
1) Clone the repo
    git clone https://github.com/aarav-singh-13/AutoJudge.git

2) Download virtual environment :
    python -m venv venv
    venv\Scripts\activate

3) Download libraries from requirements.txt
    pip install -r requirements.txt

4) Run app.py :
    ./venv/Scripts/python app.py

## Web Interface 

User gets three tet boxes to give the problem, input and output description
On clicking predict button, a post request is sent to the backend. The trained model..



# AutoJudge
Predicting Programming Problem Difficulty


## Dataset used : 
https://github.com/AREEG94FAHAD/TaskComplexityEval-24
(80% to train, 20% to test)

## Approach and Models used :

1) Pre-processing : <br>
   Removed extra spaces, all links, converted to lowercase and handled cases where input is not a string. <br>
   All textual fields were concatenated into a single field <br>
                   
3) Feature Extraction : <br>
    Used TF-IDF to monitor keywords <br>
    Hard-coded features like algebraic symbols and bitwise operators, algorithms, constraints etc <br>
4) Models : <br>
    LinearSVC <br>
    Linear Regression (See report) <br>

## Evaluation Metrics :
    Accuracy : 48%
    MSE : 12.929295363912711
    RMSE : 3.59573293835

## Steps to run locally :

1) Clone the repo :
    git clone https://github.com/aarav-singh-13/AutoJudge.git

2) Download virtual environment :
    python -m venv venv
    venv\Scripts\activate

3) Download libraries from requirements.txt :
    pip install -r requirements.txt

4) Run app.py :
    ./venv/Scripts/python app.py or python app.py

## Web Interface 

User gets three text boxes to give the problem, input and output description
On clicking predict button, a post request is sent to the backend, which returns the predicted score and class.



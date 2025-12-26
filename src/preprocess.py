import pandas as pd
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text

def preprocess_text(path):
    df = pd.read_json(path, lines=True)
    
    df["full_text"] = (
        df["title"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["input_description"].fillna("") + " " +
        df["output_description"].fillna("")
    )
    df["full_text"] = df["full_text"].apply(clean_text)
    return df


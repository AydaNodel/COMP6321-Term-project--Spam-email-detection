"""
utility script to evaluate a trained spam classification model on new text samples
loads a saved bert or distilbert model from disk, predicts labels for given inputs,
and saves the results to /src/results or specified output path
"""

import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model_path, texts):
    """
    loads a trained model and predicts spam/ham labels for given text inputs

    Args:
        model_path (str): path to the saved model directory
        texts (list): list of text strings to evaluate

    Returns:
        pd.DataFrame: dataframe containing texts and their predicted labels
    """

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # tokenize input texts
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=128
    ).to(device)

    # make predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

    # map predictions to labels
    labels = ["HAM" if p == 0 else "SPAM" for p in predictions]

    # return as dataframe
    return pd.DataFrame({"text": texts, "prediction": labels})


def save_results(results_df, output_path):
    """
    saves prediction results to a CSV file

    Args:
        results_df (pd.DataFrame): dataframe with 'text' and 'prediction' columns
        output_path (str): path to save the CSV file
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")



if __name__ == "__main__":
    # path to trained model (update as needed)
    MODEL_PATH = "src/models/final_bert_spam_classifier"
    OUTPUT_PATH = "src/evaluate/results/bert_predictions.csv"

    
    sample_texts = [
        "Congratulations! You have won a $1000 gift card. Click here to claim it.",
        "Hey Sarah, just checking if you're free for dinner tomorrow.",
        "Your bank account is on hold. Verify your details immediately.",
        "Please find attached the latest meeting notes."
    ]

    # evaluate and save
    results = evaluate_model(MODEL_PATH, sample_texts)
    print(results)
    save_results(results, OUTPUT_PATH)

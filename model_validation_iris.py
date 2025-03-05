import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import sys
import os

# Function to load validation data (for example purposes, using sklearn's load_iris dataset)
from sklearn.datasets import load_iris
def load_validation_data():
    data = load_iris()
    return data.data, data.target

# Function to validate model
def validate_model(model_uri):
    # Load the model from the MLflow registry
    model = mlflow.pyfunc.load_model(model_uri)

    # Load validation data
    X_val, y_val = load_validation_data()

    # Predict with the model
    y_pred = model.predict(X_val)

    # Evaluate performance metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')   

    # Log classification report (optional, as a summary string)
    clf_report = classification_report(y_val, y_pred, output_dict=True)
        
    # Print out the evaluation metrics
    print(f"Model Accuracy: {accuracy}")
    print(f"Model Precision: {precision}")
    print(f"Model Recall: {recall}")
    print(f"model f1: {f1}")

    # Optionally log metrics to MLflow (optional step)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_dict(clf_report, "classification_report.json")
    # Check if model meets performance criteria (e.g., accuracy > 0.9)
    if accuracy < 0.9:
        print("Model failed validation criteria. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    run_id='0cd5e4783cae4db7be2359bf75b65fbc'
    model_uri=f'runs:/{run_id}/model'
    validate_model(model_uri)

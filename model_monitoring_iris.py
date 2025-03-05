import mlflow
import pandas as pd
import time
from sklearn.metrics import accuracy_score

# Simulate fetching real-time prediction data (this could be data from your production environment)
def get_live_data():
    data = {
        'feature1': [5.1, 4.9, 4.7, 4.6],
        'feature2': [3.5, 3.0, 3.2, 3.1],
        'feature3': [1.4, 1.4, 1.3, 1.5],
        'feature4': [0.2, 0.2, 0.2, 0.2]
    }
    df = pd.DataFrame(data)
    return df

def monitor_model_performance(model_uri):
    # Load the model from MLflow model registry
    model = mlflow.pyfunc.load_model(model_uri)

    while True:
        # Fetch live data
        live_data = get_live_data()

        # Make predictions with the model
        predictions = model.predict(live_data)

        # Simulate ground truth labels for comparison (usually these would be fetched from your data store)
        true_labels = [0, 1, 0, 1]  # Example labels

        # Calculate performance metrics
        accuracy = accuracy_score(true_labels, predictions)
        print(f"Model accuracy: {accuracy}")
        # If performance drops below threshold, trigger retraining or alert
        if accuracy < 0.85:
            print("Model performance has dropped. Retraining required.")
            # Optionally, trigger an alert or retrain the model (e.g., through an API or another script)
            break

        # Wait for a specific time before the next check (e.g., 1 hour)
        time.sleep(3600)

if __name__ == "__main__":
    run_id='0cd5e4783cae4db7be2359bf75b65fbc'
    model_uri=f'runs:/{run_id}/model'
    monitor_model_performance(model_uri)

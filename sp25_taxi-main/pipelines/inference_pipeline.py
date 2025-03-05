import os
import sys
import logging
from datetime import timedelta

import pandas as pd

# Adjust sys.path so that Python can locate the src folder.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

# Now imports from src should work.
import src.config as config
from src.inference import get_feature_store, get_model_predictions, load_model_from_registry
from src.data_utils import transform_ts_data_info_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    # Get the current datetime in UTC.
    current_date = pd.Timestamp.now(tz="Etc/UTC")
    
    # Connect to the feature store.
    feature_store = get_feature_store()

    # Define the data fetching window.
    fetch_data_to = current_date - timedelta(hours=1)
    fetch_data_from = current_date - timedelta(days=29)
    logger.info(f"Fetching data from {fetch_data_from} to {fetch_data_to}")

    # Get the feature view.
    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION
    )

    # Retrieve batch time-series data, extending the window slightly to ensure completeness.
    ts_data = feature_view.get_batch_data(
        start_time=(fetch_data_from - timedelta(days=1)),
        end_time=(fetch_data_to + timedelta(days=1))
    )

    # Filter the data to the desired time window and sort it.
    ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]
    ts_data = ts_data.sort_values(["pickup_location_id", "pickup_hour"]).reset_index(drop=True)
    ts_data["pickup_hour"] = ts_data["pickup_hour"].dt.tz_localize(None)

    # Transform the raw time-series data into features.
    features = transform_ts_data_info_features(ts_data, window_size=24 * 28, step_size=23)

    # Load the model from the registry.
    model = load_model_from_registry()

    # Generate predictions using the loaded model.
    predictions = get_model_predictions(model, features)
    predictions["pickup_hour"] = current_date.ceil("h")
    
    # Log the top 30 predictions (sorted by pickup_hour descending).
    logger.info("Predictions (top 30):")
    logger.info(predictions.sort_values(by=["pickup_hour"], ascending=False).head(30))

    # Get or create the feature group for model predictions.
    feature_group = feature_store.get_or_create_feature_group(
        name=config.FEATURE_GROUP_MODEL_PREDICTION,
        version=1,
        description="Predictions from LGBM Model",
        primary_key=["pickup_location_id", "pickup_hour"],
        event_time="pickup_hour"
    )

    # Insert the predictions into the feature group.
    feature_group.insert(predictions, write_options={"wait_for_job": False})
    logger.info("Predictions inserted into the feature group successfully.")

if __name__ == "_main_":
    main()
from datetime import datetime, timedelta, timezone
import sys
import os

# Ensure Python recognizes `src/`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hopsworks
import numpy as np
import pandas as pd
from hsfs.feature_store import FeatureStore

import src.config as config
from src.data_utils import transform_ts_data_info_features


def get_hopsworks_project() -> hopsworks.project.Project:
    """Login to Hopsworks and return the project object."""
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME, api_key_value=config.HOPSWORKS_API_KEY
    )


def get_feature_store() -> FeatureStore:
    """Return the Hopsworks feature store."""
    project = get_hopsworks_project()
    return project.get_feature_store()


def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    """Predict demand for taxi pickups using a trained model."""
    predictions = model.predict(features)

    results = pd.DataFrame()
    results["pickup_location_id"] = features["pickup_location_id"].values
    results["predicted_demand"] = predictions.round(0)

    return results


def load_batch_of_features_from_store(current_date: datetime) -> pd.DataFrame:
    """Load a batch of features from the feature store."""
    feature_store = get_feature_store()

    # Define the time range
    fetch_data_to = current_date - timedelta(hours=1)
    fetch_data_from = current_date - timedelta(days=29)

    print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")

    try:
        feature_view = feature_store.get_feature_view(
            name=config.FEATURE_VIEW_NAME, version=config.FEATURE_VIEW_VERSION
        )

        ts_data = feature_view.get_batch_data(
            start_time=(fetch_data_from - timedelta(days=1)),
            end_time=(fetch_data_to + timedelta(days=1)),
        )

        ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]
        ts_data.sort_values(by=["pickup_location_id", "pickup_hour"], inplace=True)

        # Transform features
        features = transform_ts_data_info_features(
            ts_data, window_size=24 * 28, step_size=23
        )

        return features

    except Exception as e:
        print(f"❌ Error loading batch data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure


def load_model_from_registry(version=None):
    """Load the latest model from Hopsworks model registry."""
    from pathlib import Path
    import joblib
    from src.pipeline_utils import TemporalFeatureEngineer, average_rides_last_4_weeks

    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    models = model_registry.get_models(name=config.MODEL_NAME)

    if not models:
        raise ValueError("❌ No models found in the registry.")

    model = max(models, key=lambda model: model.version)
    model_dir = model.download()
    model = joblib.load(Path(model_dir) / "lgb_model.pkl")

    return model


def load_metrics_from_registry(version=None):
    """Fetch training metrics from the latest registered model."""
    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    models = model_registry.get_models(name=config.MODEL_NAME)

    if not models:
        raise ValueError("❌ No models found in the registry.")

    model = max(models, key=lambda model: model.version)
    return model.training_metrics


def fetch_next_hour_predictions():
    """Retrieve predictions for the next hour."""
    now = datetime.now(timezone.utc)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTION, version=1)

    try:
        df = fg.read()
        df = df[df["pickup_hour"] == next_hour]

        print(f"✅ Found {len(df)} records for next hour: {next_hour}")
        return df
    except Exception as e:
        print(f"❌ Error fetching predictions: {e}")
        return pd.DataFrame()


def fetch_predictions(hours):
    """Fetch predictions for the last `hours` hours."""
    current_hour = (pd.Timestamp.now(tz="Etc/UTC") - timedelta(hours=hours)).floor("h")

    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTION, version=1)

    try:
        df = fg.filter(fg.pickup_hour >= current_hour).read()
        return df
    except Exception as e:
        print(f"❌ Error fetching past predictions: {e}")
        return pd.DataFrame()


def fetch_hourly_rides(hours):
    """Fetch hourly ride data for the past `hours` hours."""
    current_hour = (pd.Timestamp.now(tz="Etc/UTC") - timedelta(hours=hours)).floor("h")

    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=1)

    try:
        query = fg.select_all().filter(fg.pickup_hour >= current_hour)
        return query.read()
    except Exception as e:
        print(f"❌ Error fetching hourly rides: {e}")
        return pd.DataFrame()


def fetch_days_data(days):
    """Fetch ride data for the past `days` days."""
    current_date = pd.to_datetime(datetime.now(timezone.utc))
    fetch_data_from = current_date - timedelta(days=(365 + days))
    fetch_data_to = current_date - timedelta(days=365)

    print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")

    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=1)

    try:
        query = fg.select_all()
        df = query.read()

        cond = (df["pickup_hour"] >= fetch_data_from) & (df["pickup_hour"] <= fetch_data_to)
        return df[cond]

    except Exception as e:
        print(f"❌ Error fetching days data: {e}")
        return pd.DataFrame()
import sys
import os
import joblib
import logging
from hsml.model_schema import ModelSchema
from hsml.schema import Schema
from sklearn.metrics import mean_absolute_error

# Ensure Python recognizes the src/ folder
current_dir = os.path.dirname(__file__)  # e.g. /path/to/this_script
project_root = os.path.abspath(os.path.join(current_dir, ".."))  # go up one level
sys.path.append(project_root)

import src.config as config
from src.data_utils import transform_ts_data_info_features_and_target
from src.inference import (
    fetch_days_data,
    get_hopsworks_project,
    load_metrics_from_registry,
    load_model_from_registry,
)
from src.pipeline_utils import get_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

try:
    # Step 1: Fetch Data
    logger.info("📊 Fetching data from group store ...")
    ts_data = fetch_days_data(180)

    if ts_data.empty:
        raise ValueError("⚠ No data fetched from the feature store!")

    logger.info(f"✅ Data fetched. Number of records: {len(ts_data)}")

    # Step 2: Transform Data
    logger.info("🔄 Transforming data into features & targets ...")
    features, targets = transform_ts_data_info_features_and_target(
        ts_data, window_size=24 * 28, step_size=23
    )

    if features.empty or targets.empty:
        raise ValueError("⚠ Transformation resulted in an empty dataset!")

    logger.info(f"✅ Transformation complete. Feature shape: {features.shape}")

    # Step 3: Initialize Model Pipeline
    pipeline = get_pipeline()

    # Step 4: Train Model
    logger.info("🛠 Training model ...")
    pipeline.fit(features, targets)

    # Step 5: Evaluate Model
    predictions = pipeline.predict(features)
    test_mae = mean_absolute_error(targets, predictions)
    metric = load_metrics_from_registry()

    logger.info(f"📉 The new MAE is {test_mae:.4f}")
    logger.info(f"📈 The previous MAE was {metric['test_mae']:.4f}")

    # Step 6: Register Model if It's Better
    if test_mae < metric.get("test_mae"):
        logger.info("🚀 Registering new model ...")
        model_path = config.MODELS_DIR / "lgb_model.pkl"
        joblib.dump(pipeline, model_path)

        input_schema = Schema(features)
        output_schema = Schema(targets)
        model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

        project = get_hopsworks_project()
        model_registry = project.get_model_registry()

        model = model_registry.sklearn.create_model(
            name="taxi_demand_predictor_next_hour",
            metrics={"test_mae": test_mae},
            input_example=features.sample(),
            model_schema=model_schema,
        )
        model.save(model_path)
        logger.info("✅ Model registered successfully!")
    else:
        logger.info("⚠ Skipping model registration because new model is not better!")

except Exception as e:
    logger.error(f"❌ Error occurred: {e}")
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import mlflow
import dagshub
import json
from pathlib import Path
from mlflow import MlflowClient
import logging


# create logger
logger = logging.getLogger("register_model")
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
logger.addHandler(handler)

# create a fomratter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# add formatter to handler
handler.setFormatter(formatter)

# initialize dagshub
import dagshub
import mlflow.client
dagshub.init(repo_owner='Aman-Husain-123', 
             repo_name='swiggy_delivery_time_prediction', 
             mlflow=True)

# set the mlflow tracking server
mlflow.set_tracking_uri("https://dagshub.com/Aman-Husain-123/swiggy_delivery_time_prediction.mlflow")


def load_model_information(file_path):
    with open(file_path) as f:
        run_info = json.load(f)
        
    return run_info


if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    
    # run information file path
    run_info_path = root_path / "run_information.json"
    
    # register the model
    run_info = load_model_information(run_info_path)
    
    # get the run id
    run_id = run_info["run_id"]
    model_name = run_info["model_name"]
    
    # model to register path
    model_registry_path = f"runs:/{run_id}/{model_name}"
    
    
    # register the model with a retry mechanism for DagsHub eventual consistency
    import time
    max_retries = 10
    retry_delay = 10
    model_version = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to register model (Attempt {attempt + 1}/{max_retries})...")
            model_version = mlflow.register_model(model_uri=model_registry_path,
                                                name=model_name)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Registration failed: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Failed to register model after multiple attempts.")
                raise e
    
    
    # get the model version
    registered_model_version = model_version.version
    registered_model_name = model_version.name
    logger.info(f"The latest model version in model registry is {registered_model_version}")
    
    # update the stage of the model to staging
    client = MlflowClient()
    client.transition_model_version_stage(
        name=registered_model_name,
        version=registered_model_version,
        stage="Staging"
    )
    
    logger.info("Model pushed to Staging stage")
    
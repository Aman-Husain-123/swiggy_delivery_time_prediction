import pandas as pd
import yaml
import joblib
import logging
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge

TARGET = "time_taken"

# create logger
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
logger.addHandler(handler)

# create a formatter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error("The file to load does not exist")
        raise
    return df


def read_params(file_path: Path):
    with open(file_path, "r") as f:
        params_file = yaml.safe_load(f)
    return params_file


def make_X_and_y(data: pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent

    # data load path
    train_data_path = root_path / "data" / "processed" / "train_trans.csv"
    
    # params path
    params_file_path = root_path / "params.yaml"
    
    # models save dir
    models_save_dir = root_path / "models"
    models_save_dir.mkdir(exist_ok=True, parents=True)

    # load training data
    train_df = load_data(train_data_path)
    logger.info("Processed training data loaded successfully")

    # split into X and y
    X_train, y_train = make_X_and_y(train_df, TARGET)

    # read parameters
    params = read_params(params_file_path)["Train"]
    rf_params = params["Random_Forest"]
    lgb_params = params["LightGBM"]
    logger.info("Parameters loaded successfully")

    # instantiate base estimators
    rf = RandomForestRegressor(random_state=42, **rf_params)
    lgb = LGBMRegressor(random_state=42, **lgb_params)

    # stacking regressor
    stacking_regressor = StackingRegressor(
        estimators=[
            ("rf", rf),
            ("lgb", lgb)
        ],
        final_estimator=Ridge()
    )

    # power transformer for target
    power_transformer = PowerTransformer()

    # transformed target regressor
    model = TransformedTargetRegressor(
        regressor=stacking_regressor,
        transformer=power_transformer
    )

    # train model
    logger.info("Training stacking regressor model...")
    model.fit(X_train, y_train)
    logger.info("Model training complete")

    # save artifacts
    model_save_path = models_save_dir / "model.joblib"
    pt_save_path = models_save_dir / "power_transformer.joblib"
    sr_save_path = models_save_dir / "stacking_regressor.joblib"

    joblib.dump(model, model_save_path)
    joblib.dump(model.transformer_, pt_save_path)
    joblib.dump(model.regressor_, sr_save_path)
    logger.info("All model artifacts saved successfully")
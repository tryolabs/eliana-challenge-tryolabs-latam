import os
import pickle
import logging
from datetime import datetime
from typing import Tuple, Union, List

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from google.cloud import storage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOCAL_MODEL_PATH = "challenge/models/model.pkl"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
THRESHOLD_IN_MINUTES = 15
GCS_MODEL_PATH = "model.pkl"
GCS_BUCKET = os.getenv("GCS_BUCKET")


class DelayModel:
    def __init__(self):
        self._model = self._load_model(LOCAL_MODEL_PATH)
        self.top_10_features = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air",
        ]

    def _load_model(self, path: str) -> Union[None, object]:
        """
        Load the model from a pickle file.

        Args:
            path (str): Path to the model file.

        Returns:
            object: Loaded model or None if the file does not exist.
        """
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    model = pickle.load(f)
                    logger.info("Model loaded from local folder.")
                    return model
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return None
        if GCS_BUCKET:
            return self._load_model_from_gcs(GCS_BUCKET)
        logger.warning("Model file not found")

    def _load_model_from_gcs(self, bucket_name: str) -> Union[None, object]:
        """
        Load the model from a Google Cloud Storage bucket.

        Args:
            bucket_name (str): Name of the bucket.

        Returns:
            object: Loaded model or None if the file does not exist.
        """
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(GCS_MODEL_PATH)
        if blob.exists():
            try:
                with blob.open("rb") as f:
                    model = pickle.load(f)
                    logger.info("Model loaded from GCS bucket.")
                    return model
            except Exception as e:
                logger.error(f"Error loading model from GCS: {e}")
                return None
        logger.warning("Model file not found in GCS bucket")
        return None

    def _get_min_diff(self, data: pd.Series) -> float:
        """
        Calculate the difference in minutes between 'Fecha-O' and 'Fecha-I'.

        Args:
            data (pd.Series): Row of the DataFrame.

        Returns:
            float: Difference in minutes.
        """
        try:
            fecha_o = datetime.strptime(data["Fecha-O"], DATE_FORMAT)
            fecha_i = datetime.strptime(data["Fecha-I"], DATE_FORMAT)
            min_diff = (fecha_o - fecha_i).total_seconds() / 60
            return min_diff
        except ValueError as e:
            logger.error(f"Date parsing error: {e}")
            return np.nan

    def _is_delayed(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add a new column 'delay' to the DataFrame indicating if the flight is
        delayed.
        Args:
            data (pd.DataFrame): Raw data.
        Returns:
            pd.DataFrame: DataFrame with the new column 'delay'.
        """
        data["min_diff"] = data.apply(self._get_min_diff, axis=1)
        data["delay"] = np.where(data["min_diff"] > THRESHOLD_IN_MINUTES, 1, 0)
        return data

    def _fill_missing_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing features with default values.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            pd.DataFrame: data with all required features.
        """
        missing_features = set(self.top_10_features) - set(features.columns)
        for feature in missing_features:
            features[feature] = 0
        return features

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        if not all(
            col in data.columns for col in ["OPERA", "TIPOVUELO", "MES"]
        ):
            raise ValueError("Required columns are missing in the input data.")
        features = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1,
        )
        features = self._fill_missing_features(features)
        features = features[self.top_10_features]
        if target_column:
            data = self._is_delayed(data)
            target = data[[target_column]]
            return features, target
        return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            features, target["delay"], test_size=0.33, random_state=42
        )
        n_y0 = len(y_train[y_train == 0])
        n_y1 = len(y_train[y_train == 1])
        scale = n_y0 / n_y1
        print(scale)
        xgb_model = xgb.XGBClassifier(
            random_state=1, learning_rate=0.01, scale_pos_weight=scale
        )
        self._model = xgb_model.fit(x_train, y_train)
        logger.info("Model trained.")
        # Log some performance inforamation
        y_pred = self._model.predict(x_test)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        logger.info("Model performance: \n%s", report)
        logger.info("Confusion matrix: \n%s", cm)
        # Save the model
        with open(LOCAL_MODEL_PATH, "wb") as f:
            pickle.dump(self._model, f)
        logger.info(f"Model saved at {LOCAL_MODEL_PATH}")
        return

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        predictions = self._model.predict(features)
        return predictions.tolist()

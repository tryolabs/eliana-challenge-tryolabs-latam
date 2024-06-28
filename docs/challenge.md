# Software Engineer (ML & LLMs) Challenge - Documentation

## Introduction
This document provides an overview of the steps taken to complete the Software Engineer (ML & LLMs) Challenge.

## Part I: Model Transcription

### Model Selection
The Data Scientist (DS) suggested using either XGBoost or Logistic Regression trained with top 10 features and class balancing, which was in good agreement with the findings they shared.

In comparing XGBoost and Logistic Regression for predicting delayed flights, both models demonstrated similar performance metrics. Despite this parity in performance, I decided to select XGBoost after careful consideration of several factors.

#### XGBoost

XGBoost was chosen primarily due to its robustness and flexibility, which make it highly suitable for real-world scenarios involving large datasets and complex feature sets. Specifically, XGBoost offers:

- **High Accuracy and Efficiency:** XGBoost is known for its superior performance in handling large volumes of data and intricate features, often resulting in higher accuracy and efficiency,  though this can vary depending on the specific dataset and problem.
- **Handling of Missing Values and Outliers:** The model has built-in mechanisms to manage missing values effectively, which is crucial in dealing with messy, real-world data. While it does not specifically handle outliers, its tree-based nature can mitigate their impact.
- **Hyperparameter Tuning:** XGBoost provides extensive options for hyperparameter tuning, allowing for future improvements and fine-tuning to enhance model performance.
- **Feature Importance Insights:** The model offers valuable insights into feature importance through methods like SHAP values, though it is generally less interpretable than Logistic Regression.

#### Logistic Regression

While XGBoost was ultimately selected, Logistic Regression remains a strong contender due to its unique advantages:

- **Simplicity and Interpretability:** Logistic Regression is simpler to implement and easier to interpret, offering clear insights into the relationships between features and the target variable.
- **Computational Efficiency:** It is computationally efficient and performs well with smaller datasets or when computational resources are limited.
- **Reduced Overfitting:** Logistic Regression is less prone to overfitting, particularly when dealing with a smaller number of features and applying regularization techniques.


### Conclusion

Both XGBoost and Logistic Regression demonstrated similar performance metrics in predicting flight delays. The decision to choose XGBoost was driven by its robustness, flexibility, and suitability for handling more complex datasets and scenarios. However, the merits of Logistic Regression—such as its simplicity, efficiency, and interpretability—underscore that the choice was not trivial. Each model has its own set of strengths, and the decision was made with a forward-looking perspective on future enhancements and the ability to manage more intricate data challenges.

**NOTE**: I was instructed not to make any improvements to the model. Therefore, I focused on selecting a model using the information provided by the DS and will discuss potential improvements in the following section.

### Improvements and Fixes

Upon reviewing the provided Jupyter notebook from the Data Scientist (DS), several areas for improvement and potential fixes were identified. However, due to the instruction not to alter the model and the need to pass the provided tests, most of these issues were noted but not addressed in the current implementation:

1. **Coding Bugs**:
   - **Unused Variable**: The notebook defined a `training_data` variable that was never used in subsequent steps. This could be removed in the future to clean up the code and avoid potential confusion.
   - **Visualization Bug**: There was a bug related to `sns.barplot` that hindered data visualization. This was addressed to ensure that the data could be visualized correctly using Seaborn's bar plot functionality.

2. **Feature Naming Consistency**:
   - **Mixed Languages**: Feature names were inconsistently written in both English and Spanish. Additionally, the format varied with some names in lowercase and others in uppercase. Standardizing all feature names to a single language (preferably English) and using a consistent format (e.g., all lowercase) would ensure uniformity and avoid potential issues during data preprocessing and model training.

3. **Feature Selection**:
   - **Arbitrary Subset**: Before training for the first time an arbitrary subset of features was selected without explicit justification. A more systematic approach to feature selection could be employed in the future to ensure that the most relevant features are used.
   - **Feature Importance Discrepancy**: When performing feature importance analysis, the top 10 features identified were not consistently used in the final model. Ensuring that the top 10 features from the feature importance analysis are accurately selected and used in the model training process would enhance the model's robustness.
   - **Hardcoded Features**: The feature selection process in the notebook involved hardcoding specific features, which is not a best practice. Future implementations should dynamically select features based on their importance or other criteria to ensure flexibility and scalability. This hardcoding was transcribed into `model.py` as it was needed to pass the provided tests.

4. **Other Improvements**:
   - **Hyperparameter Tuning**: Although instructed not to make improvements to the model, it was noted that the notebook lacked a comprehensive hyperparameter tuning process. Implementing a robust grid search or randomized search for hyperparameter optimization could significantly enhance model performance.
   -**Local Model Storage**: For testing purposes, a local version of the model was saved. This practice can be useful for debugging and quick iterations, but in a production environment, a more robust model versioning and storage solution should be implemented.


The identified issues represent opportunities for future improvements to ensure a more robust and justifiable model training process.


## Part II: API Development
The model was deployed using FastAPI as specified. The key endpoints implemented are:
- `/`: Root endpoint providing a welcome message.
- `/health`: Health check endpoint to verify the API status.
- `/predict`: Accepts flight details and returns the probability of delay.

### Input Validation
Input validators were implemented to ensure that:
- `MES` is an integer between 1 and 12.
- `OPERA` is one of the predefined airlines.
- `TIPOVUELO` is either 'N' (National) or 'I' (International).

The API was tested using the provided test cases, and it passed all of them successfully.

## Part III: Deployment
The API was deployed on Google Cloud Platform (GCP) using the following steps:
1. **Containerization**: Used Docker to containerize the FastAPI application.
2. **Image Storage**: Stored the Docker image in Google Container Registry (GCR).
3. **Deployment**: Deployed the containerized application to Google Cloud Run.

The API is accessible at this [url](https://flight-delay-api-wi6l66iubq-uc.a.run.app), which is also updated in the Makefile.

The API passed the provided stress test successfully.

## Part IV: CI/CD Implementation
A CI/CD pipeline was created using GitHub Actions. The steps included:

### Continuous Integration (CI)

The CI workflow (`ci.yml`) includes:
1. **Trigger**: Runs on pushes to `feature/*`, `hotfix/*`, `release/*`, `develop`, and `main` branches, and on pull requests to `develop` and `main` branches.
2. **Jobs**:
   - **Testing**:
     - Runs model tests.
     - Runs API tests.

### Continuous Delivery (CD)

The CD workflow (`cd.yml`) includes:
1. **Trigger**: Runs on pushes to the `main` branch.
2. **Jobs**:
   - **Building and Pushing Docker Image**: Builds and pushes the Docker image to Google Container Registry (GCR).
   - **Deployment**: Deploys the container to Google Cloud Run.
   - **Post-Deployment**: Runs a stress test using the deployed API URL.

#### Model Versioning

For the CD flow, a version of the model was required to be available. The model was uploaded to a Google Cloud Storage (GCS) bucket for this purpose. This was a practical approach to accomplish the CI/CD workflow for the challenge but would not be the preferred choice in a real-world scenario.

#### Better Practices for Model Versioning
While using GCS for model storage was practical for this challenge, better practices for model versioning and management include:
- **Model Registry**: Tools like MLflow provide a dedicated model registry to track model versions, metadata, and lineage.
- **Automated Model Deployment**: Integrating a model registry with CI/CD pipelines can automate the deployment of specific model versions, ensuring consistency and reproducibility.


## Conclusion
This challenge offered a comprehensive experience in operationalizing a machine learning model, deploying it via an API, and implementing CI/CD pipelines. The steps taken ensure that the model is not only accurate but also scalable and maintainable.While the current implementation meets the challenge requirements, several areas for improvement have been identified throughout the process and discussed in each section.

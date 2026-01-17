# src/pipeline.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


# ---------------------------
# Feature Selection Function
# ---------------------------
def get_feature_columns():
    """
    Returns the list of final features used for training.
    These features are safe from data leakage.
    """
    return [
        'gender_male',
        'race/ethnicity_group B',
        'race/ethnicity_group C',
        'race/ethnicity_group D',
        'race/ethnicity_group E',
        "parental level of education_bachelor's degree",
        'parental level of education_high school',
        "parental level of education_master's degree",
        'parental level of education_some college',
        'parental level of education_some high school',
        'lunch_standard',
        'test preparation course_none'
    ]


# ---------------------------
# Build Training Pipeline
# ---------------------------
def build_pipeline():
    """
    Builds and returns a machine learning pipeline.
    """

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )

    pipeline = Pipeline(
        steps=[
            ('model', model)
        ]
    )

    return pipeline


# ---------------------------
# Train Pipeline
# ---------------------------
def train_pipeline(df, target='result_binary'):
    """
    Trains the pipeline and returns the trained model.
    """

    feature_cols = get_feature_columns()

    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    print(f"Validation F1 Score: {f1:.4f}")

    return pipeline

"""
Model Training Module
Random Forest Classifier for protein function prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.dummy import DummyClassifier
import joblib
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_features, get_feature_names

# Set random seed for reproducibility
np.random.seed(42)
RANDOM_STATE = 42


def load_data(csv_path: str) -> tuple:
    """
    Load protein data from CSV and extract features.
    
    Args:
        csv_path: Path to proteins.csv
        
    Returns:
        Tuple of (X features, y labels, label encoder)
    """
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    print(f"Total samples: {len(df)}")
    print(f"\nClass distribution:")
    print(df["label"].value_counts())
    
    # Extract features
    print("\nExtracting features...")
    X_list = []
    y_list = []
    valid_count = 0
    
    for idx, row in df.iterrows():
        features = extract_features(row["sequence"])
        if features is not None:
            X_list.append(features)
            y_list.append(row["label"])
            valid_count += 1
        
        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1}/{len(df)} sequences...")
    
    print(f"\nValid samples: {valid_count}/{len(df)}")
    
    X = np.vstack(X_list)
    y = np.array(y_list)
    
    return X, y


def plot_class_distribution(y: np.ndarray, title: str, save_path: str):
    """Plot and save class distribution."""
    unique, counts = np.unique(y, return_counts=True)
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(unique, counts, color=['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'])
    plt.xlabel('Functional Class')
    plt.ylabel('Count')
    plt.title(title)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(count), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrix(cm: np.ndarray, classes: list, save_path: str):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_feature_importance(model: RandomForestClassifier, feature_names: list, save_path: str, top_n: int = 15):
    """Plot and save feature importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), importances[indices][::-1], color='#3498db')
    plt.yticks(range(top_n), [feature_names[i] for i in indices][::-1])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def train_model(X: np.ndarray, y: np.ndarray, output_dir: str = "models") -> dict:
    """
    Train Random Forest classifier.
    
    Args:
        X: Feature matrix
        y: Labels
        output_dir: Directory to save model and plots
        
    Returns:
        Dictionary with results
    """
    print("\n" + "=" * 50)
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print("=" * 50)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot class distribution (before split)
    plot_class_distribution(
        y, "Class Distribution (Full Dataset)",
        os.path.join(plots_dir, "class_distribution.png")
    )
    
    # STRATIFIED split (critical fix from review)
    print("\nSplitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    classes = model.classes_
    
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"\nAccuracy: {accuracy:.2%}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Random baseline comparison
    print("\n" + "-" * 50)
    print("BASELINE COMPARISON")
    print("-" * 50)
    
    dummy = DummyClassifier(strategy='stratified', random_state=RANDOM_STATE)
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    dummy_accuracy = accuracy_score(y_test, dummy_pred)
    
    print(f"Random baseline accuracy: {dummy_accuracy:.2%}")
    print(f"Model accuracy: {accuracy:.2%}")
    print(f"Improvement over random: {(accuracy - dummy_accuracy):.2%}")
    
    if accuracy > dummy_accuracy:
        print("✓ Model performs better than random guessing!")
    else:
        print("✗ Warning: Model does not beat random baseline")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        cm, classes.tolist(),
        os.path.join(plots_dir, "confusion_matrix.png")
    )
    
    # Plot feature importance
    feature_names = get_feature_names()
    plot_feature_importance(
        model, feature_names,
        os.path.join(plots_dir, "feature_importance.png")
    )
    
    # Save model
    model_path = os.path.join(output_dir, "classifier.joblib")
    joblib.dump({
        'model': model,
        'classes': classes,
        'feature_names': feature_names,
        'accuracy': accuracy
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    return {
        'model': model,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classes': classes,
        'baseline_accuracy': dummy_accuracy
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train protein function classifier")
    parser.add_argument("--data", default="data/proteins.csv", help="Path to training data")
    parser.add_argument("--output", default="models", help="Output directory")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation only")
    args = parser.parse_args()
    
    # Load data
    X, y = load_data(args.data)
    
    # Train model
    results = train_model(X, y, args.output)
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Final accuracy: {results['accuracy']:.2%}")
    print(f"Baseline: {results['baseline_accuracy']:.2%}")

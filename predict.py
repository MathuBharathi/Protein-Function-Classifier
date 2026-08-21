"""
Protein Function Prediction Tool
Main entry point for predictions.
"""

import numpy as np
import joblib
import os
import sys
from typing import Dict, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from features import extract_features, clean_sequence, AMINO_ACIDS

# Default model path
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "classifier.joblib")


def validate_sequence(sequence: str) -> tuple:
    """
    Validate protein sequence.
    
    Args:
        sequence: Raw protein sequence
        
    Returns:
        Tuple of (is_valid, error_message, cleaned_sequence)
    """
    if not sequence:
        return False, "Empty sequence provided", ""
    
    # Clean sequence
    cleaned = clean_sequence(sequence)
    
    if len(cleaned) == 0:
        return False, "No valid amino acids found in sequence", ""
    
    if len(cleaned) < 50:
        return False, f"Sequence too short ({len(cleaned)} AA). Minimum 50 required.", ""
    
    if len(cleaned) > 10000:
        return False, f"Sequence too long ({len(cleaned)} AA). Maximum 10000 allowed.", ""
    
    # Check for invalid characters
    invalid_chars = set(sequence.upper()) - set(AMINO_ACIDS) - set(' \n\t\r')
    if invalid_chars:
        return False, f"Contains invalid characters: {invalid_chars}", ""
    
    return True, "", cleaned


def load_model(model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """
    Load trained model from disk.
    
    Args:
        model_path: Path to saved model file
        
    Returns:
        Dictionary with model and metadata
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please run: python src/model.py --data data/proteins.csv"
        )
    
    return joblib.load(model_path)


def predict_function(sequence: str, model_path: str = DEFAULT_MODEL_PATH) -> Dict:
    """
    Predict protein functional class from sequence.
    
    This is the main AI-assisted prediction tool.
    
    Args:
        sequence: Protein amino acid sequence (single letter codes)
        model_path: Path to trained model (optional)
        
    Returns:
        Dictionary with:
            - predicted_class: The predicted functional class
            - confidence: Dictionary of probabilities for each class
            - sequence_length: Length of cleaned sequence
            - is_valid: Whether prediction was successful
            - error: Error message if prediction failed
    """
    # Validate input
    is_valid, error_msg, cleaned_seq = validate_sequence(sequence)
    
    if not is_valid:
        return {
            "predicted_class": None,
            "confidence": {},
            "sequence_length": 0,
            "is_valid": False,
            "error": error_msg
        }
    
    try:
        # Load model
        model_data = load_model(model_path)
        model = model_data['model']
        classes = model_data['classes']
        
        # Extract features
        features = extract_features(cleaned_seq)
        
        if features is None:
            return {
                "predicted_class": None,
                "confidence": {},
                "sequence_length": len(cleaned_seq),
                "is_valid": False,
                "error": "Feature extraction failed"
            }
        
        # Reshape for prediction
        X = features.reshape(1, -1)
        
        # Predict
        predicted_class = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Build confidence dictionary
        confidence = {cls: round(float(prob), 4) for cls, prob in zip(classes, probabilities)}
        
        # Sort by probability (descending)
        confidence = dict(sorted(confidence.items(), key=lambda x: x[1], reverse=True))
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "sequence_length": len(cleaned_seq),
            "is_valid": True,
            "error": None
        }
        
    except Exception as e:
        return {
            "predicted_class": None,
            "confidence": {},
            "sequence_length": len(cleaned_seq) if cleaned_seq else 0,
            "is_valid": False,
            "error": str(e)
        }


def format_result(result: Dict) -> str:
    """
    Format prediction result for display.
    
    Args:
        result: Output from predict_function()
        
    Returns:
        Formatted string
    """
    if not result["is_valid"]:
        return f"❌ Prediction Failed: {result['error']}"
    
    output = []
    output.append("=" * 50)
    output.append("🧬 PROTEIN FUNCTION PREDICTION")
    output.append("=" * 50)
    output.append(f"\nSequence Length: {result['sequence_length']} amino acids")
    output.append(f"\n✅ Predicted Class: {result['predicted_class']}")
    output.append("\nConfidence Scores:")
    output.append("-" * 30)
    
    for cls, prob in result["confidence"].items():
        bar = "█" * int(prob * 20)
        output.append(f"  {cls:12s}: {prob:.2%} {bar}")
    
    output.append("=" * 50)
    
    return "\n".join(output)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict protein function from sequence")
    parser.add_argument("sequence", nargs="?", help="Protein sequence (or use --file)")
    parser.add_argument("--file", "-f", help="Read sequence from file")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL_PATH, help="Path to model")
    args = parser.parse_args()
    
    # Get sequence
    if args.file:
        with open(args.file, 'r') as f:
            sequence = f.read()
    elif args.sequence:
        sequence = args.sequence
    else:
        # Interactive mode
        print("Enter protein sequence (press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        sequence = "".join(lines)
    
    # Predict
    result = predict_function(sequence, args.model)
    print(format_result(result))

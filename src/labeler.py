"""
Protein Labeler
Maps protein keywords/function to functional classes.
"""

import pandas as pd
import re
from typing import Optional


# Keyword patterns for each class (case-insensitive)
CLASS_PATTERNS = {
    "Enzyme": [
        r"kinase", r"synthase", r"dehydrogenase", r"protease", r"oxidase",
        r"reductase", r"transferase", r"hydrolase", r"ligase", r"isomerase",
        r"lyase", r"phosphatase", r"ATPase", r"catalytic", r"enzymatic"
    ],
    "Binding": [
        r"binding", r"receptor", r"antibody", r"ligand", r"immunoglobulin",
        r"antigen", r"DNA-binding", r"RNA-binding", r"calcium-binding"
    ],
    "Transporter": [
        r"transporter", r"channel", r"pump", r"carrier", r"symporter",
        r"antiporter", r"permease", r"exchanger", r"ion channel"
    ],
    "Regulatory": [
        r"transcription", r"regulator", r"repressor", r"activator",
        r"transcription factor", r"regulatory", r"modulator", r"inhibitor"
    ]
}

# Terms indicating unknown/hypothetical function - SKIP THESE
UNKNOWN_PATTERNS = [
    r"hypothetical", r"uncharacterized", r"putative", r"probable",
    r"unknown function", r"predicted", r"unnamed"
]


def classify_protein(keywords: str, function_text: str = "") -> Optional[str]:
    """
    Classify a protein based on its keywords and function description.
    
    Args:
        keywords: Semicolon-separated keywords from UniProt
        function_text: Function description text
        
    Returns:
        Class label or None if cannot be classified
    """
    # Combine keywords and function for matching
    combined_text = f"{keywords} {function_text}".lower()
    
    # Check for unknown/hypothetical proteins
    for pattern in UNKNOWN_PATTERNS:
        if re.search(pattern, combined_text, re.IGNORECASE):
            return None  # Will be filtered out
    
    # Score each class based on pattern matches
    class_scores = {}
    
    for class_name, patterns in CLASS_PATTERNS.items():
        score = 0
        for pattern in patterns:
            matches = len(re.findall(pattern, combined_text, re.IGNORECASE))
            score += matches
        class_scores[class_name] = score
    
    # Get the class with highest score
    max_score = max(class_scores.values())
    
    if max_score == 0:
        return None  # No matching patterns - UNKNOWN
    
    # Return the class with highest score
    for class_name, score in class_scores.items():
        if score == max_score:
            return class_name
    
    return None


def label_dataset(input_csv: str, output_csv: str) -> pd.DataFrame:
    """
    Label proteins in a CSV file.
    
    Args:
        input_csv: Path to input CSV with keywords column
        output_csv: Path to save labeled CSV
        
    Returns:
        Labeled DataFrame
    """
    df = pd.read_csv(input_csv)
    
    # Apply classification
    df["label"] = df.apply(
        lambda row: classify_protein(
            str(row.get("keywords", "")),
            str(row.get("function", ""))
        ),
        axis=1
    )
    
    # Remove UNKNOWN (None) labels
    original_count = len(df)
    df = df.dropna(subset=["label"])
    filtered_count = original_count - len(df)
    
    print(f"Filtered out {filtered_count} unknown/unclear proteins")
    
    # Save labeled data
    df.to_csv(output_csv, index=False)
    
    print(f"\nClass distribution:")
    print(df["label"].value_counts())
    
    return df


def balance_classes(df: pd.DataFrame, min_samples: int = 200) -> pd.DataFrame:
    """
    Balance classes by downsampling majority classes.
    
    Args:
        df: DataFrame with 'label' column
        min_samples: Minimum samples per class
        
    Returns:
        Balanced DataFrame
    """
    # Find minimum class size (but at least min_samples if possible)
    class_counts = df["label"].value_counts()
    target_size = min(class_counts.min(), max(class_counts.values))
    
    # Ensure we have at least min_samples if data allows
    if target_size < min_samples:
        print(f"Warning: Some classes have fewer than {min_samples} samples")
    
    # Downsample each class
    balanced_dfs = []
    for label in df["label"].unique():
        class_df = df[df["label"] == label]
        if len(class_df) > target_size:
            class_df = class_df.sample(n=target_size, random_state=42)
        balanced_dfs.append(class_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    print(f"\nBalanced class distribution:")
    print(balanced_df["label"].value_counts())
    
    return balanced_df


if __name__ == "__main__":
    # Example usage
    import os
    
    data_dir = "data"
    if os.path.exists(os.path.join(data_dir, "proteins_full.csv")):
        df = label_dataset(
            os.path.join(data_dir, "proteins_full.csv"),
            os.path.join(data_dir, "proteins_labeled.csv")
        )
        
        balanced = balance_classes(df)
        balanced[["sequence", "label"]].to_csv(
            os.path.join(data_dir, "proteins.csv"),
            index=False
        )

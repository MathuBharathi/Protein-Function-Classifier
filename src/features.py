"""
Feature Extraction Module
Extracts 24 numerical features from protein sequences.
"""

import numpy as np
from typing import Dict, List, Optional

# Standard amino acids
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# Molecular weights of amino acids (Daltons)
AA_MOLECULAR_WEIGHTS = {
    'A': 89.09, 'C': 121.15, 'D': 133.10, 'E': 147.13, 'F': 165.19,
    'G': 75.07, 'H': 155.16, 'I': 131.17, 'K': 146.19, 'L': 131.17,
    'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.20,
    'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19
}

# Kyte-Doolittle hydrophobicity scale (for GRAVY score)
HYDROPATHY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

# pK values for isoelectric point calculation
PK_VALUES = {
    'N_term': 9.69,  # N-terminus
    'C_term': 2.34,  # C-terminus
    'D': 3.86, 'E': 4.25,  # Acidic
    'C': 8.33, 'Y': 10.07,  # Neutral with pK
    'H': 6.00, 'K': 10.54, 'R': 12.48  # Basic
}


def clean_sequence(sequence: str) -> str:
    """
    Clean and validate protein sequence.
    
    Args:
        sequence: Raw protein sequence
        
    Returns:
        Cleaned uppercase sequence with only valid AAs
    """
    # Uppercase and remove whitespace
    sequence = sequence.upper().strip()
    sequence = ''.join(sequence.split())
    
    # Keep only valid amino acids
    valid_sequence = ''.join([aa for aa in sequence if aa in AMINO_ACIDS])
    
    return valid_sequence


def amino_acid_composition(sequence: str) -> Dict[str, float]:
    """
    Calculate normalized amino acid composition.
    
    Args:
        sequence: Cleaned protein sequence
        
    Returns:
        Dictionary with frequency of each amino acid (sum = 1.0)
    """
    length = len(sequence)
    if length == 0:
        return {aa: 0.0 for aa in AMINO_ACIDS}
    
    composition = {}
    for aa in AMINO_ACIDS:
        count = sequence.count(aa)
        composition[aa] = count / length  # NORMALIZED
    
    return composition


def molecular_weight(sequence: str) -> float:
    """
    Calculate molecular weight in Daltons.
    
    Args:
        sequence: Cleaned protein sequence
        
    Returns:
        Molecular weight
    """
    if len(sequence) == 0:
        return 0.0
    
    # Sum of AA weights minus water released per peptide bond
    water_weight = 18.015
    mw = sum(AA_MOLECULAR_WEIGHTS.get(aa, 0) for aa in sequence)
    mw -= (len(sequence) - 1) * water_weight
    
    return mw


def _charge_at_pH(sequence: str, pH: float) -> float:
    """Calculate net charge at given pH."""
    positive = 0.0
    negative = 0.0
    
    # N-terminus
    positive += 1.0 / (1.0 + 10**(pH - PK_VALUES['N_term']))
    
    # C-terminus
    negative += 1.0 / (1.0 + 10**(PK_VALUES['C_term'] - pH))
    
    # Charged residues
    for aa in sequence:
        if aa in ['K', 'R', 'H']:
            positive += 1.0 / (1.0 + 10**(pH - PK_VALUES[aa]))
        elif aa in ['D', 'E']:
            negative += 1.0 / (1.0 + 10**(PK_VALUES[aa] - pH))
        elif aa == 'C':
            negative += 1.0 / (1.0 + 10**(PK_VALUES[aa] - pH))
        elif aa == 'Y':
            negative += 1.0 / (1.0 + 10**(PK_VALUES[aa] - pH))
    
    return positive - negative


def isoelectric_point(sequence: str) -> float:
    """
    Calculate isoelectric point (pI) using bisection method.
    
    Args:
        sequence: Cleaned protein sequence
        
    Returns:
        Isoelectric point (pH where net charge = 0)
    """
    if len(sequence) == 0:
        return 7.0  # Neutral default
    
    pH_min, pH_max = 0.0, 14.0
    
    # Bisection method
    for _ in range(100):
        pH_mid = (pH_min + pH_max) / 2.0
        charge = _charge_at_pH(sequence, pH_mid)
        
        if abs(charge) < 0.001:
            return pH_mid
        
        if charge > 0:
            pH_min = pH_mid
        else:
            pH_max = pH_mid
    
    return (pH_min + pH_max) / 2.0


def gravy_score(sequence: str) -> float:
    """
    Calculate GRAVY (Grand Average of Hydropathy) score.
    
    Expected range: approximately -2 to +2
    
    Args:
        sequence: Cleaned protein sequence
        
    Returns:
        GRAVY score
    """
    if len(sequence) == 0:
        return 0.0
    
    total = sum(HYDROPATHY.get(aa, 0) for aa in sequence)
    gravy = total / len(sequence)
    
    # Sanity check for expected range
    if gravy < -4.5 or gravy > 4.5:
        print(f"Warning: GRAVY score {gravy} outside expected range")
    
    return gravy


def extract_features(sequence: str) -> Optional[np.ndarray]:
    """
    Extract all 24 features from a protein sequence.
    
    Features (24 total):
        - Amino acid composition: 20 features (normalized)
        - Sequence length: 1 feature
        - Molecular weight: 1 feature
        - Isoelectric point: 1 feature
        - GRAVY score: 1 feature
    
    Args:
        sequence: Raw protein sequence
        
    Returns:
        numpy array of 24 features, or None if sequence is invalid
    """
    # Clean sequence
    seq = clean_sequence(sequence)
    
    # Validate minimum length
    if len(seq) < 50:
        return None
    
    features = []
    
    # 1. Amino acid composition (20 features)
    aa_comp = amino_acid_composition(seq)
    for aa in AMINO_ACIDS:
        features.append(aa_comp[aa])
    
    # 2. Sequence length (1 feature)
    features.append(len(seq))
    
    # 3. Molecular weight (1 feature)
    features.append(molecular_weight(seq))
    
    # 4. Isoelectric point (1 feature)
    features.append(isoelectric_point(seq))
    
    # 5. GRAVY score (1 feature)
    features.append(gravy_score(seq))
    
    return np.array(features)


def get_feature_names() -> List[str]:
    """
    Get names of all 24 features.
    
    Returns:
        List of feature names
    """
    names = []
    
    # Amino acid composition
    for aa in AMINO_ACIDS:
        names.append(f"AA_{aa}")
    
    # Other features
    names.extend(["seq_length", "mol_weight", "isoelectric_point", "gravy_score"])
    
    return names


def extract_features_batch(sequences: List[str]) -> np.ndarray:
    """
    Extract features for multiple sequences.
    
    Args:
        sequences: List of protein sequences
        
    Returns:
        2D numpy array of shape (n_sequences, 24)
    """
    features_list = []
    valid_indices = []
    
    for i, seq in enumerate(sequences):
        feat = extract_features(seq)
        if feat is not None:
            features_list.append(feat)
            valid_indices.append(i)
        else:
            print(f"Warning: Skipping invalid sequence at index {i}")
    
    if len(features_list) == 0:
        return np.array([])
    
    return np.vstack(features_list)


if __name__ == "__main__":
    # Test with example sequence
    test_seq = "MKTFFVLVLLALVAGATQAGEADAVYLDNGIKLGTPAVTRSGEDVVTYDGALHEVFTSPEFSLHLGP"
    
    print("Feature Extraction Test")
    print("=" * 50)
    print(f"Sequence length: {len(test_seq)}")
    
    features = extract_features(test_seq)
    names = get_feature_names()
    
    print(f"\nExtracted {len(features)} features:")
    print("-" * 50)
    
    for name, value in zip(names, features):
        print(f"{name:20s}: {value:.4f}")
    
    # Verify AA composition sums to 1.0
    aa_sum = sum(features[:20])
    print(f"\nAA composition sum: {aa_sum:.4f} (should be ~1.0)")
    
    # Verify GRAVY in expected range
    gravy = features[-1]
    print(f"GRAVY score: {gravy:.4f} (expected range: -2 to +2)")

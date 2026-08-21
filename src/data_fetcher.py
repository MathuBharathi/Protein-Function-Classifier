"""
UniProt Data Fetcher
Downloads reviewed Swiss-Prot protein entries with sequences and keywords.
"""

import requests
import pandas as pd
import time
import os
from typing import Optional

# UniProt REST API endpoint
UNIPROT_API = "https://rest.uniprot.org/uniprotkb/search"

# Keywords for each functional class (used for targeted queries)
CLASS_KEYWORDS = {
    "Enzyme": ["kinase", "synthase", "dehydrogenase", "protease", "oxidase", "reductase", "transferase", "hydrolase"],
    "Binding": ["binding", "receptor", "antibody", "ligand-binding"],
    "Transporter": ["transporter", "channel", "pump", "carrier", "symporter", "antiporter"],
    "Regulatory": ["transcription", "regulator", "repressor", "activator", "transcription factor"]
}

# Target proteins per class
TARGET_PER_CLASS = 300


def fetch_proteins_for_class(class_name: str, keywords: list, max_results: int = 300) -> list:
    """
    Fetch proteins from UniProt for a specific functional class.
    
    Args:
        class_name: Name of the functional class
        keywords: List of keywords to search for
        max_results: Maximum number of proteins to fetch
        
    Returns:
        List of protein dictionaries with sequence, keywords, function
    """
    proteins = []
    
    for keyword in keywords:
        if len(proteins) >= max_results:
            break
            
        # Build query for reviewed Swiss-Prot entries
        query = f'(keyword:"{keyword}") AND (reviewed:true) AND (length:[100 TO 1000])'
        
        params = {
            "query": query,
            "format": "json",
            "fields": "accession,sequence,keyword,protein_name,cc_function",
            "size": min(100, max_results - len(proteins))
        }
        
        try:
            print(f"  Fetching '{keyword}' for {class_name}...")
            response = requests.get(UNIPROT_API, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            for entry in results:
                # Skip entries with vague descriptions
                function_text = ""
                if "comments" in entry:
                    for comment in entry.get("comments", []):
                        if comment.get("commentType") == "FUNCTION":
                            texts = comment.get("texts", [])
                            if texts:
                                function_text = texts[0].get("value", "")
                
                protein_name = entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "")
                
                # Skip hypothetical/uncharacterized proteins
                skip_terms = ["hypothetical", "uncharacterized", "putative", "probable", "unknown function"]
                if any(term.lower() in (function_text + protein_name).lower() for term in skip_terms):
                    continue
                
                # Extract keywords
                keywords_list = [kw.get("name", "") for kw in entry.get("keywords", [])]
                
                protein = {
                    "accession": entry.get("primaryAccession", ""),
                    "sequence": entry.get("sequence", {}).get("value", ""),
                    "keywords": "; ".join(keywords_list),
                    "function": function_text,
                    "protein_name": protein_name,
                    "class": class_name
                }
                
                # Only add if sequence exists and is valid
                if protein["sequence"] and len(protein["sequence"]) >= 50:
                    proteins.append(protein)
                    
            # Rate limiting
            time.sleep(0.5)
            
        except requests.RequestException as e:
            print(f"  Warning: Failed to fetch '{keyword}': {e}")
            continue
    
    print(f"  → Fetched {len(proteins)} proteins for {class_name}")
    return proteins[:max_results]


def fetch_all_proteins(output_dir: str = "data") -> pd.DataFrame:
    """
    Fetch proteins for all functional classes and save to CSV.
    
    Args:
        output_dir: Directory to save the CSV file
        
    Returns:
        DataFrame with all proteins
    """
    all_proteins = []
    
    print("=" * 50)
    print("UniProt Data Fetcher - Protein Function Classifier")
    print("=" * 50)
    
    for class_name, keywords in CLASS_KEYWORDS.items():
        print(f"\n[{class_name}]")
        proteins = fetch_proteins_for_class(class_name, keywords, TARGET_PER_CLASS)
        all_proteins.extend(proteins)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_proteins)
    
    # Remove duplicates by accession
    df = df.drop_duplicates(subset=["accession"])
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full data
    full_path = os.path.join(output_dir, "proteins_full.csv")
    df.to_csv(full_path, index=False)
    print(f"\nSaved full data to: {full_path}")
    
    # Save simplified version (sequence, label)
    simple_df = df[["sequence", "class"]].rename(columns={"class": "label"})
    simple_path = os.path.join(output_dir, "proteins.csv")
    simple_df.to_csv(simple_path, index=False)
    print(f"Saved training data to: {simple_path}")
    
    # Print class distribution
    print("\n" + "=" * 50)
    print("Class Distribution:")
    print("=" * 50)
    print(df["class"].value_counts())
    print(f"\nTotal proteins: {len(df)}")
    
    return df


if __name__ == "__main__":
    # Run data fetching
    df = fetch_all_proteins()

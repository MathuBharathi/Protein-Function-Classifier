# 🧬 Protein Function Classifier

**AI-Assisted Tool for Predicting Protein Functional Classes from Sequences**

> Hackathon project: Machine learning approach to classify proteins into broad functional categories.

---

## 🎯 Problem

Gene function annotation is slow and expensive. Manual curation of protein databases can't keep pace with sequencing output. This tool provides a computational first-pass classification.

## 📊 Classes Predicted

| Class | Description | Keywords |
|-------|-------------|----------|
| **Enzyme** | Catalytic proteins | kinase, synthase, dehydrogenase |
| **Binding** | Ligand/receptor binding | receptor, antibody, binding |
| **Transporter** | Membrane transport | channel, pump, carrier |
| **Regulatory** | Transcription control | transcription factor, regulator |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Data (First Time)

```bash
python src/data_fetcher.py
```

This downloads ~1000 reviewed proteins from UniProt Swiss-Prot.

### 3. Train Model

```bash
python src/model.py --data data/proteins.csv
```

### 4. Make Predictions

```bash
python predict.py "MKTFFVLVLLALVAGATQAGEADAVYLDNGI..."
```

Or use the Jupyter notebook:

```bash
jupyter notebook demo.ipynb
```

---

## 📁 Project Structure

```
xn v thon/
├── data/                    # Training data
│   └── proteins.csv         # Sequences + labels
├── src/
│   ├── data_fetcher.py      # UniProt data download
│   ├── labeler.py           # Keyword → class mapping
│   ├── features.py          # Feature extraction (24 features)
│   └── model.py             # Random Forest training
├── models/
│   ├── classifier.joblib    # Saved model
│   └── plots/               # Visualization outputs
├── predict.py               # CLI prediction tool
├── demo.ipynb               # Interactive demo
└── README.md                # This file
```

---

## 🔬 Method

### Features (24 total)

| Feature | Count | Description |
|---------|-------|-------------|
| Amino acid composition | 20 | Frequency of each AA (normalized) |
| Sequence length | 1 | Number of residues |
| Molecular weight | 1 | Sum of AA molecular weights |
| Isoelectric point | 1 | pH where net charge = 0 |
| GRAVY score | 1 | Grand Average of Hydropathy |

### Model

- **Algorithm**: Random Forest Classifier (100 trees)
- **Split**: 80% train / 20% test (stratified)
- **Output**: Class prediction + probability scores

---

## 📈 Results

| Metric | Value |
|--------|-------|
| **Model Accuracy** | ≥60% (target) |
| **Random Baseline** | ~25% |
| **Improvement** | +35%+ |

See `models/plots/` for:
- Confusion matrix
- Feature importance
- Class distribution

---

## ⚠️ Limitations

1. **Broad classes only** - Cannot predict fine-grained GO terms
2. **Sequence-based features only** - No structural information
3. **Not a replacement for wet lab** - Computational prediction only
4. **Limited to reviewed proteins** - May not generalize to novel sequences

---

## 💻 Usage Examples

### Python API

```python
from predict import predict_function

result = predict_function("MKTFFVLVLLALVAGATQAGE...")

print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']}")
```

### Command Line

```bash
# Direct sequence
python predict.py "ACDEFGHIKLMNPQRSTVWY..."

# From file
python predict.py --file protein.fasta
```

---

## 📚 Data Source

- **Database**: UniProt Swiss-Prot (reviewed)
- **Entries**: ~1000 proteins
- **Quality**: Manually curated, high confidence

---

## 🛠️ Tech Stack

- Python 3.9+
- scikit-learn (Random Forest)
- BioPython (sequence analysis)
- pandas, numpy, matplotlib

---

*Built for Hackathon 2026*

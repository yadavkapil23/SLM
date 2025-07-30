# Small Language Model (SLM) Project

A complete pipeline for training a small transformer-based language model using TensorFlow/Keras. This project downloads the WikiText-103 dataset, trains a Byte-Level BPE tokenizer, converts data to TFRecord format, and trains a TinyTransformer model.

## 📁 Project Structure

```
slm_project/
├── data/
│   ├── corpus.txt          # Raw training text (497MB)
│   └── dataset.tfrecord    # Tokenized training data (16GB)
├── tokenizer/
│   ├── vocab.json          # BPE vocabulary (462KB)
│   └── merges.txt          # Merge rules for tokenizer (267KB)
├── model/
│   └── tiny_transformer.py # Transformer model definition
├── scripts/
│   ├── train_tokenizer.py  # Trains tokenizer from corpus.txt
│   ├── write_tfrecord.py   # Converts tokenized data to TFRecord
│   └── count_tfrecord.py   # Utility to count TFRecord examples
├── DATA_Conversion.py      # Downloads and prepares corpus.txt
├── train.py               # Main training script
├── requirements.txt        # Python dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download and Prepare Data

```bash
python DATA_Conversion.py
```

This will:
- Download the WikiText-103-raw-v1 dataset
- Create `data/corpus.txt` (497MB)

### 3. Train Tokenizer

```bash
python scripts/train_tokenizer.py
```

This will:
- Train a ByteLevelBPETokenizer on `corpus.txt`
- Create `tokenizer/vocab.json` and `tokenizer/merges.txt`

### 4. Create TFRecord Dataset

```bash
python scripts/write_tfrecord.py
```

This will:
- Tokenize the corpus using the trained tokenizer
- Create `data/dataset.tfrecord` (16GB, 29.9M examples)

### 5. Train the Model

```bash
python train.py
```

This will:
- Load the TFRecord dataset
- Train the TinyTransformer model
- Save the trained model as `trained_slm_model/`

## 📋 Detailed File Descriptions

### Data Preparation

#### `DATA_Conversion.py`
- **Purpose**: Downloads WikiText-103 dataset and creates `corpus.txt`
- **Output**: `data/corpus.txt` (497MB, ~1.1M lines)
- **Features**: 
  - Downloads from Hugging Face datasets
  - Filters empty lines
  - Progress indicators

#### `scripts/train_tokenizer.py`
- **Purpose**: Trains ByteLevelBPETokenizer on corpus
- **Input**: `data/corpus.txt`
- **Output**: `tokenizer/vocab.json`, `tokenizer/merges.txt`
- **Features**:
  - Vocabulary size: 30,000 tokens
  - Error handling for missing files

#### `scripts/write_tfrecord.py`
- **Purpose**: Converts tokenized data to TFRecord format
- **Input**: `data/corpus.txt`, `tokenizer/` files
- **Output**: `data/dataset.tfrecord` (16GB, 29.9M examples)
- **Features**:
  - Sequence length: 128 tokens
  - Error handling for missing files
  - Progress tracking

### Model Architecture

#### `model/tiny_transformer.py`
- **Purpose**: Defines the TinyTransformer model
- **Architecture**:
  - Embedding layer (vocab_size → embed_dim)
  - Sinusoidal positional encoding
  - Multi-head attention
  - Feed-forward network
  - Layer normalization
  - Residual connections
- **Parameters**:
  - Vocabulary size: 30,000
  - Embedding dimension: 128
  - Attention heads: 2
  - Feed-forward dimension: 512
  - Max sequence length: 512

### Training

#### `train.py`
- **Purpose**: Main training script
- **Features**:
  - Loads TFRecord dataset
  - Splits into train/validation (90%/10%)
  - Early stopping (patience=3)
  - Learning rate reduction (patience=2)
  - Model checkpointing
- **Training Parameters**:
  - Epochs: 5 (max)
  - Batch size: 32
  - Learning rate: 1e-4
  - Sequence length: 128

## 🔧 Configuration

### Model Parameters (in `train.py`)
```python
vocab_size = 30000
seq_len = 128
embed_dim = 128
num_heads = 2
ff_dim = 512
batch_size = 32
epochs = 5
```

### Tokenizer Parameters (in `scripts/train_tokenizer.py`)
```python
vocab_size = 30000
```

## 📊 Dataset Statistics

- **Raw Corpus**: 497MB, ~1.1M lines
- **Tokenized Dataset**: 16GB, 29,909,128 examples
- **Training Examples**: ~26,918,215 (90%)
- **Validation Examples**: ~2,990,913 (10%)

## 🛠️ Dependencies

```
tensorflow>=2.10.0
datasets>=2.0.0
tokenizers>=0.13.0
```

## 📈 Training Progress

The training includes:
- **Early Stopping**: Stops if no improvement for 3 epochs
- **Learning Rate Reduction**: Reduces LR by 50% if no improvement for 2 epochs
- **Validation Monitoring**: Tracks accuracy and loss on validation set
- **Model Checkpointing**: Saves best model weights

## 🎯 Expected Outputs

After running the complete pipeline, you'll have:
- ✅ `data/corpus.txt` - Raw training text
- ✅ `tokenizer/vocab.json` - BPE vocabulary
- ✅ `tokenizer/merges.txt` - Merge rules
- ✅ `data/dataset.tfrecord` - Tokenized training data
- ✅ `trained_slm_model/` - Saved trained model

## ⚠️ Notes

- **Memory Requirements**: The TFRecord creation step requires significant RAM (~8GB+)
- **Processing Time**: 
  - Tokenizer training: ~1-2 minutes
  - TFRecord creation: ~30-45 minutes
  - Model training: Depends on hardware (typically 1-4 hours)
- **Storage**: Total project size ~17GB (mostly TFRecord file)

## 🔍 Troubleshooting

### Common Issues:
1. **Out of Memory**: Reduce batch_size in `train.py`
2. **Missing Files**: Ensure previous steps completed successfully
3. **Slow Processing**: TFRecord creation is CPU-intensive, be patient

### File Verification:
```bash
# Check corpus size
ls -lh data/corpus.txt

# Check tokenizer files
ls -lh tokenizer/

# Check TFRecord size
ls -lh data/dataset.tfrecord
```

## 🎉 Success Indicators

- Corpus file: ~497MB
- Tokenizer files: ~729KB total
- TFRecord file: ~16GB
- Training: Model saves successfully to `trained_slm_model/`

---

**Happy Training! 🚀**

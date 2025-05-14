First, create a virtual environment and install the required dependencies:

```bash
# Create a virtual environment
python -m venv tia-env
source tia-env/bin/activate  # On Windows: tia-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Run the following commands to download and preprocess the datasets:
# Create directories
mkdir -p data/raw data/processed

# Download datasets
python data/download_datasets.py

# Preprocess datasets
python data/preprocessing/preprocess.py
python data/preprocessing/preprocess-WikiSQL.py
python data/preprocessing/preprocess-Mturk.py

├── figure_eight/
│   ├── train.json
│   ├── val.json
│   ├── test.json
│   └── metadata.json
├── wikisql/
│   ├── train.json
│   ├── val.json
│   ├── test.json
│   └── metadata.json
└── mturk/
    ├── train.json
    ├── val.json
    ├── test.json
    └── metadata.json
We provide pre-trained Swin Transformer weights as a starting point. Download them using
python pretrained_weights/download_weights.py
Training New Models (Figure Eight)
python experiments/train.py \
    --config experiments/configs/figure_eight_config.json \
    --output_dir models/figure_eight \
    --log_dir logs/figure_eight
Training Parameters
The key hyperparameters used in our experiments are:

Learning rate: 3e-5
Batch size: 32
Number of epochs: 50
Weight decay: 0.01
Optimizer: AdamW

These parameters can be adjusted in the respective config files.

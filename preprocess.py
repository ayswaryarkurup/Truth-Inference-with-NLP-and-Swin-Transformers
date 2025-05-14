# Create the preprocess_figure_eight.py file
cat > data/preprocessing/preprocess_figure_eight.py << 'EOF'
# Figure Eight dataset.
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import random

def load_raw_data(input_path):
    """
    Load raw Figure Eight dataset.
    
    Args:
        input_path: Path to the raw data
        
    Returns:
        DataFrame containing the raw data
    """
    print(f"Loading raw data from {input_path}...")
        
    try:
        df = pd.read_csv(os.path.join(input_path, "figure_eight_raw.csv"))
        print(f"Loaded {len(df)} entries from Figure Eight dataset")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def extract_tasks_and_contributors(df):
    #Extract tasks and contributor information from raw data.
   
    tasks = {}
    contributors = {}
    labels = {}
    
    print("Extracting tasks and contributors...")
        
    for _, row in tqdm(df.iterrows(), total=len(df)):
        task_id = row.get('task_id', str(random.randint(10000, 99999)))
        contributor_id = row.get('contributor_id', str(random.randint(1000, 9999)))
        
        # Extract task information
        if task_id not in tasks:
            tasks[task_id] = {
                'description': row.get('task_description', 'Example task description'),
                'difficulty': row.get('task_difficulty', random.random()),
                'true_label': row.get('true_label', random.randint(0, 1))
            }
        
        # Extract contributor information
        if contributor_id not in contributors:
            contributors[contributor_id] = {
                'reliability': row.get('contributor_reliability', random.random()),
                'experience': row.get('contributor_experience', random.randint(1, 5))
            }
        
        # Extract label information
        if task_id not in labels:
            labels[task_id] = {}
        
        labels[task_id][contributor_id] = row.get('label', random.randint(0, 1))
    
    print(f"Extracted {len(tasks)} tasks and {len(contributors)} contributors")
    return tasks, contributors, labels

def split_data(tasks, test_ratio=0.2, val_ratio=0.1):
    task_ids = list(tasks.keys())
    random.shuffle(task_ids)
    
    test_size = int(len(task_ids) * test_ratio)
    val_size = int(len(task_ids) * val_ratio)
    
    test_ids = task_ids[:test_size]
    val_ids = task_ids[test_size:test_size + val_size]
    train_ids = task_ids[test_size + val_size:]
    
    print(f"Split data into {len(train_ids)} train, {len(val_ids)} validation, and {len(test_ids)} test tasks")
    return train_ids, val_ids, test_ids

def create_processed_data(tasks, contributors, labels, task_ids, split_name):
    
    processed_data = {
        'tasks': [],
        'contributors': list(contributors.keys()),
        'label_space': [0, 1]  # Binary classification for this example
    }
    
    for task_id in task_ids:
        task_data = tasks[task_id]
        task_contributors = list(labels[task_id].keys())
        task_labels = [labels[task_id][c] for c in task_contributors]
        
        processed_data['tasks'].append({
            'task_id': task_id,
            'description': task_data['description'],
            'difficulty': task_data['difficulty'],
            'true_label': task_data['true_label'],
            'contributors': task_contributors,
            'labels': task_labels
        })
    
    return processed_data

def preprocess_figure_eight(input_path, output_path):
   
    # Load raw data
    df = load_raw_data(input_path)
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Extract information
    tasks, contributors, labels = extract_tasks_and_contributors(df)
    
    # Split data
    train_ids, val_ids, test_ids = split_data(tasks)
    
    # Create processed data for each split
    train_data = create_processed_data(tasks, contributors, labels, train_ids, 'train')
    val_data = create_processed_data(tasks, contributors, labels, val_ids, 'val')
    test_data = create_processed_data(tasks, contributors, labels, test_ids, 'test')
    
    # Create metadata
    metadata = {
        'dataset_name': 'Figure Eight',
        'num_tasks': len(tasks),
        'num_contributors': len(contributors),
        'label_space': [0, 1],
        'train_size': len(train_ids),
        'val_size': len(val_ids),
        'test_size': len(test_ids)
    }
    
    # Save processed data
    os.makedirs(output_path, exist_ok=True)
    
    with open(os.path.join(output_path, 'train.json'), 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(os.path.join(output_path, 'val.json'), 'w') as f:
        json.dump(val_data, f, indent=2)
    
    with open(os.path.join(output_path, 'test.json'), 'w') as f:
        json.dump(test_data, f, indent=2)
    
    with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess Figure Eight dataset')
    parser.add_argument('--input_path', type=str, default='data/raw/figure_eight',
                        help='Path to raw data')
    parser.add_argument('--output_path', type=str, default='data/processed/figure_eight',
                        help='Path to save processed data')
    
    args = parser.parse_args()
    
    preprocess_figure_eight(args.input_path, args.output_path)
EOF


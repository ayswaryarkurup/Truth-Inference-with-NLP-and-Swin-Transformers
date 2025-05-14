
# Create the preprocess_wikisql.py file
cat > data/preprocessing/preprocess_wikisql.py << 'EOF'

import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import random
import sqlite3

def load_raw_data(input_path):
    """
    Load raw WikiSQL dataset.
    
    Args:
        input_path: Path to the raw data
        
    Returns:
        Dictionary containing the raw data
    """
    print(f"Loading raw data from {input_path}...")    
   
    try:
        # Load train, dev, and test data
        with open(os.path.join(input_path, "train.jsonl"), 'r') as f:
            train_data = [json.loads(line) for line in f]
        
        with open(os.path.join(input_path, "dev.jsonl"), 'r') as f:
            dev_data = [json.loads(line) for line in f]
        
        with open(os.path.join(input_path, "test.jsonl"), 'r') as f:
            test_data = [json.loads(line) for line in f]
        
        
        with open(os.path.join(input_path, "tables.json"), 'r') as f:
            tables = json.load(f)
        
        raw_data = {
            'train': train_data,
            'dev': dev_data,
            'test': test_data,
            'tables': tables
        }
        
        print(f"Loaded {len(train_data)} train, {len(dev_data)} dev, and {len(test_data)} test examples")
        return raw_data
    
    except FileNotFoundError:
        print("WikiSQL files not found.")
      
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def simulate_contributor_responses(raw_data, num_contributors=300, error_rate_range=(0.05, 0.3)):
    tasks = {}
    contributors = {}
    labels = {}
    
    print("Simulating contributor responses...")
    
    # Create contributors with different reliability levels
    for i in range(num_contributors):
        contributor_id = f"contributor_{i}"
        error_rate = random.uniform(error_rate_range[0], error_rate_range[1])
        contributors[contributor_id] = {
            'reliability': 1.0 - error_rate,
            'experience': random.randint(1, 5)
        }
    
    # Process train examples as tasks
    for i, example in enumerate(tqdm(raw_data['train'], desc="Processing train examples")):
        task_id = f"task_{i}"
        question = example.get('question', f'Example question {i}')
        true_sql = example.get('sql', {}).get('query', f'SELECT * FROM table WHERE id = {i}')
        
        tasks[task_id] = {
            'description': question,
            'difficulty': random.uniform(0.1, 0.9),  # task difficulty
            'true_label': true_sql
        }
        
        
        task_contributors = random.sample(list(contributors.keys()), random.randint(10, 20))
        labels[task_id] = {}
        
        for contributor_id in task_contributors:
            contributor_reliability = contributors[contributor_id]['reliability']
            
            
            if random.random() < contributor_reliability:
                # Correct response
                sql = true_sql
            else:
                # Incorrect response
                error_type = random.choice(['syntax', 'column', 'condition', 'table'])
                if error_type == 'syntax':
                    sql = true_sql.replace('SELECT', 'SLCT')
                elif error_type == 'column':
                    sql = true_sql.replace('*', 'id, name')
                elif error_type == 'condition':
                    sql = true_sql.replace('=', '>')
                else:
                    sql = true_sql.replace('table', 'tab')
            
            labels[task_id][contributor_id] = sql
    
    #Additional tasks
    offset = len(tasks)
    for i, example in enumerate(tqdm(raw_data['dev'], desc="Processing dev examples")):
        task_id = f"task_{i + offset}"
        question = example.get('question', f'Example question {i + offset}')
        true_sql = example.get('sql', {}).get('query', f'SELECT * FROM table WHERE id = {i + offset}')
        
        tasks[task_id] = {
            'description': question,
            'difficulty': random.uniform(0.1, 0.9),
            'true_label': true_sql
        }
        
        task_contributors = random.sample(list(contributors.keys()), random.randint(10, 20))
        labels[task_id] = {}
        
        for contributor_id in task_contributors:
            contributor_reliability = contributors[contributor_id]['reliability']
            
            if random.random() < contributor_reliability:
                sql = true_sql
            else:
                error_type = random.choice(['syntax', 'column', 'condition', 'table'])
                if error_type == 'syntax':
                    sql = true_sql.replace('SELECT', 'SLCT')
                elif error_type == 'column':
                    sql = true_sql.replace('*', 'id, name')
                elif error_type == 'condition':
                    sql = true_sql.replace('=', '>')
                else:
                    sql = true_sql.replace('table', 'tab')
            
            labels[task_id][contributor_id] = sql
    
    print(f"Created {len(tasks)} tasks with contributions from {len(contributors)} contributors")
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
        'label_type': 'sql'
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

def preprocess_wikisql(input_path, output_path):
    
    # Load raw data
    raw_data = load_raw_data(input_path)
    if raw_data is None:
        print("Failed to load data. Exiting.")
        return
    
    # contributor responses
    tasks, contributors, labels = simulate_contributor_responses(raw_data)
    
    # Split data
    train_ids, val_ids, test_ids = split_data(tasks)
    
    # Create processed data for each split
    train_data = create_processed_data(tasks, contributors, labels, train_ids, 'train')
    val_data = create_processed_data(tasks, contributors, labels, val_ids, 'val')
    test_data = create_processed_data(tasks, contributors, labels, test_ids, 'test')
    
    # Create metadata
    metadata = {
        'dataset_name': 'WikiSQL',
        'num_tasks': len(tasks),
        'num_contributors': len(contributors),
        'label_type': 'sql',
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
    parser = argparse.ArgumentParser(description='Preprocess WikiSQL dataset')
    parser.add_argument('--input_path', type=str, default='data/raw/wikisql',
                        help='Path to raw data')
    parser.add_argument('--output_path', type=str, default='data/processed/wikisql',
                        help='Path to save processed data')
    
    args = parser.parse_args()
    
    preprocess_wikisql(args.input_path, args.output_path)
EOF


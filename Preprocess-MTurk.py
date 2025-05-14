
# Create the preprocess_mturk.py file
cat > data/preprocessing/preprocess_mturk.py << 'EOF'
#Preprocessing script for the MTurk dataset.

import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import random
import re
from collections import Counter

def load_raw_data(input_path):
   
    print(f"Loading raw data from {input_path}...")
      
    try:
        files = [f for f in os.listdir(input_path) if f.endswith('.csv')]
        
        if not files:
            raise FileNotFoundError("No CSV files found")
        
        dfs = []
        for file in files:
            df = pd.read_csv(os.path.join(input_path, file))
            dfs.append(df)
        
        df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(df)} entries from MTurk dataset")
        return df
    
    except FileNotFoundError:
        print("Not found")
                
        data = []
        for i in range(15000):  # 15,000 tasks as specified
            text = random.choice(texts) if i < 10 else f" #{i}"
            true_sentiment = random.choice(sentiments)
            
            #  5-10 MTurk workers per task
            for j in range(random.randint(5, 10)):
                worker_id = f"worker_{random.randint(1, 400)}" 
                
                # Workers have 80% chance of providing correct sentiment
                if random.random() < 0.8:
                    sentiment = true_sentiment
                else:
                    sentiment = random.choice([s for s in sentiments if s != true_sentiment])
                
                data.append({
                    'task_id': f"task_{i}",
                    'text': text,
                    'worker_id': worker_id,
                    'sentiment': sentiment,
                    'true_sentiment': true_sentiment,
                    'worker_experience': random.randint(1, 5),
                    'task_timestamp': pd.Timestamp('2023-01-01') + pd.Timedelta(days=random.randint(0, 365))
                })
        
        df = pd.DataFrame(data)
        print(f" {len(df)} worker responses across {df['task_id'].nunique()} tasks")
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def extract_tasks_and_contributors(df):
    
    tasks = {}
    contributors = {}
    labels = {}
    
    print("Extracting tasks and contributors...")
    
    # Group by task_id to get unique tasks
    task_groups = df.groupby('task_id')
    
    for task_id, group in tqdm(task_groups):
        
        first_row = group.iloc[0]
        text = first_row.get('text', 'Example text')
        
        # Determine true label from 'true_sentiment' if available, otherwise use majority vote
        if 'true_sentiment' in group.columns:
            true_label = first_row['true_sentiment']
        else:
            # Use majority voting to determine the true label
            label_counts = group['sentiment'].value_counts()
            true_label = label_counts.index[0]
        
        # Calculate difficulty based on worker agreement
        sentiments = group['sentiment'].tolist()
        most_common = Counter(sentiments).most_common(1)[0][1]
        agreement_ratio = most_common / len(sentiments)
        difficulty = 1.0 - agreement_ratio  # Higher difficulty for lower agreement
        
        tasks[task_id] = {
            'description': text,
            'difficulty': difficulty,
            'true_label': true_label
        }
        
        # Extract contributor  information
        for _, row in group.iterrows():
            worker_id = row.get('worker_id', f"worker_{random.randint(1000, 9999)}")
            
            if worker_id not in contributors:
                if 'worker_experience' in row:
                    experience = row['worker_experience']
                else:
                    experience = random.randint(1, 5)
                
                contributors[worker_id] = {
                    'experience': experience,
                    'reliability': None  
                }
            
            # Store label information
            if task_id not in labels:
                labels[task_id] = {}
            
            labels[task_id][worker_id] = row.get('sentiment', random.choice(['positive', 'negative', 'neutral']))
    
    # Calculate worker reliability 
    for worker_id in contributors:
        correct_count = 0
        total_count = 0
        
        for task_id in labels:
            if worker_id in labels[task_id]:
                total_count += 1
                if labels[task_id][worker_id] == tasks[task_id]['true_label']:
                    correct_count += 1
        
        reliability = correct_count / total_count if total_count > 0 else 0.5
        contributors[worker_id]['reliability'] = reliability
    
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
        'label_space': ['positive', 'negative', 'neutral']  # Sentiment analysis
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

def preprocess_mturk(input_path, output_path):
    
    # Load raw data
    df = load_raw_data(input_path)
    if df is None:
        print("Exiting.")
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
        'dataset_name': 'MTurk',
        'num_tasks': len(tasks),
        'num_contributors': len(contributors),
        'label_space': ['positive', 'negative', 'neutral'],
        'train_size': len(train_ids),
        'val_size': len(val_ids),
        'test_size': len(test_ids)
    }
    
    # Save processed data
    os.makedirs(output_path, exist_ok=True)
    
    with open(os.path.join(output_path, 'train.json'), 'w') as f:
        json.dump(train_data, f, indent=2

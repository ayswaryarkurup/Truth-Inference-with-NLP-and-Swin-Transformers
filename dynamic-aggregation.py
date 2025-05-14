"""
Dynamic Aggregation module for the TIA model.
This module implements the dynamic aggregation algorithm for truth inference, which iteratively refines contributor reliability scores and inferred labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DynamicAggregation(nn.Module):
    """
    Dynamic Aggregation module for truth inference.
    
    1. Evaluates contributor reliability based on their agreement with estimated true labels
    2. Dynamically updates reliability scores through an iterative process
    3. Infers true labels by weighted aggregation of contributor labels
    """
    
    def __init__(self, hidden_size=768, num_labels=2, num_contributors=None, max_iterations=10, convergence_threshold=0.001):
        
        super(DynamicAggregation, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.num_contributors = num_contributors
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # Initialize contributor reliability estimation
        self.reliability_estimation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Swish(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.Swish(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Task difficulty estimation
        self.task_difficulty_estimation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Swish(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize contributor embeddings with number of contributors
        if num_contributors is not None:
            self.contributor_embeddings = nn.Embedding(num_contributors, hidden_size // 4)
            self.contributor_reliability_bias = nn.Parameter(torch.zeros(num_contributors))
        else:
            self.contributor_embeddings = None
            self.contributor_reliability_bias = None
    
    def _initialize_reliability(self, contributor_ids, contributor_labels):
        #Initialize contributor reliability scores based on majority voting.
        
        batch_size = contributor_labels.size(0)
        num_contributors = contributor_labels.size(1)
        
        # Count votes for each label
        label_counts = torch.zeros(batch_size, self.num_labels, device=contributor_labels.device)
        for i in range(batch_size):
            for j in range(num_contributors):
                label = contributor_labels[i, j]
                if label != -1:  # Ignore missing labels
                    label_counts[i, label] += 1
        
        # Get majority vote label
        initial_labels = torch.argmax(label_counts, dim=1)
        
        # Calculate initial reliability as agreement with majority vote
        reliability_scores = torch.zeros(num_contributors, device=contributor_labels.device)
        
        for i in range(num_contributors):
            agreement_count = 0
            total_count = 0
            
            for j in range(batch_size):
                if contributor_labels[j, i] != -1:  # Ignore missing labels
                    total_count += 1
                    if contributor_labels[j, i] == initial_labels[j]:
                        agreement_count += 1
            
            if total_count > 0:
                reliability_scores[i] = agreement_count / total_count
            else:
                reliability_scores[i] = 0.5  # Default reliability if no labels provided
        
        return reliability_scores, initial_labels
    
    def forward(self, task_embeddings, contributor_ids, contributor_labels):
        #Forward pass of the Dynamic Aggregation module.
        batch_size = task_embeddings.size(0)
        num_contributors = contributor_labels.size(1)
        
        # Estimate task difficulty
        task_difficulty = self.task_difficulty_estimation(task_embeddings)
        
        # Initialize contributor reliability and labels
        reliability_scores, current_labels = self._initialize_reliability(contributor_ids, contributor_labels)
        
        # Iterative refinement
        for iteration in range(self.max_iterations):
            prev_labels = current_labels.clone()
            
            # Update reliability scores based on task embeddings and contributor IDs
            if self.contributor_embeddings is not None:
                # Get contributor embeddings
                contributor_embeds = self.contributor_embeddings(contributor_ids)
                
                # Reshape for batch processing
                contributor_embeds = contributor_embeds.view(batch_size, num_contributors, -1)
                task_embeds_expanded = task_embeddings.unsqueeze(1).expand(-1, num_contributors, -1)
                
                # Combine task and contributor information
                combined_embeds = torch.cat([task_embeds_expanded, contributor_embeds], dim=2)
                combined_embeds = combined_embeds.view(-1, self.hidden_size + self.hidden_size // 4)
                
                # Estimate reliability
                estimated_reliability = self.reliability_estimation(combined_embeds)
                estimated_reliability = estimated_reliability.view(batch_size, num_contributors)
                
                # Apply contributor-specific bias
                contributor_bias = self.contributor_reliability_bias[contributor_ids].view(batch_size, num_contributors)
                reliability_scores = torch.sigmoid(estimated_reliability + contributor_bias)
            else:
                # Use initial reliability scores if contributor embeddings are not available
                reliability_scores = reliability_scores.unsqueeze(0).expand(batch_size, -1)
            
            # Weighted voting based on reliability scores and task difficulty
            label_weights = torch.zeros(batch_size, self.num_labels, device=task_embeddings.device)
            
            for i in range(batch_size):
                for j in range(num_contributors):
                    label = contributor_labels[i, j]
                    if label != -1: 
                        weight = reliability_scores[i, j] * (1 - task_difficulty[i])
                        label_weights[i, label] += weight
            
            # Update current labels
            current_labels = torch.argmax(label_weights, dim=1)
            
            # convergence
            if torch.sum(current_labels != prev_labels) / batch_size < self.convergence_threshold:
                break
        
        return current_labels, reliability_scores
    
    def infer(self, task_embeddings, contributor_labels):
        #Infer true labels from contributor labels.
        
        batch_size = task_embeddings.size(0)
        num_contributors = contributor_labels.size(1)
        
      
        if self.contributor_embeddings is None:
            contributor_ids = torch.arange(num_contributors, device=task_embeddings.device)
            contributor_ids = contributor_ids.unsqueeze(0).expand(batch_size, -1)
        else:
            
            contributor_ids = torch.zeros(batch_size, num_contributors, device=task_embeddings.device).long()
        
        return self.forward(task_embeddings, contributor_ids, contributor_labels)
    
    def reliability_loss(self, contributor_reliability, inferred_labels, true_labels):
        # Calculate loss for contributor reliability estimation.
       
        # Calculate accuracy of inferred labels
        accuracy = (inferred_labels == true_labels).float()
        
       
        target_reliability = accuracy.unsqueeze(1).expand_as(contributor_reliability)
        
        # Mean squared error loss
        loss = F.mse_loss(contributor_reliability, target_reliability)
        
        return loss

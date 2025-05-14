import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin_transformer import SwinTransformer
from .dynamic_aggregation import DynamicAggregation

class TIAModel(nn.Module):
    """
    Truth Inference model using NLP and Swin Transformers.
    
    The model consists of three main components:
    1. Swin Transformer for contextual embeddings
    2. Dynamic aggregation algorithm for truth inference
    3. Transfer learning mechanism for domain adaptation
    """
    
    def __init__(
        self,
        vocab_size=30522,  # Default BERT vocab size
        hidden_size=768,
        num_labels=2,
        max_position_embeddings=512,
        num_attention_heads=12,
        num_hidden_layers=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        swin_window_size=7,
        swin_mlp_ratio=4.0,
        use_swish_activation=True,
        num_contributors=None,
        pretrained_model_path=None,
    ):
        
        super(TIAModel, self).__init__()
        
        # Swin Transformer for contextual embeddings
        self.swin_transformer = SwinTransformer(
            img_size=224,  # Treated as sequence length after encoding
            patch_size=4,  # Size of patches
            in_chans=1,    # Single channel input (text)
            num_classes=hidden_size,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=swin_window_size,
            mlp_ratio=swin_mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=hidden_dropout_prob,
            attn_drop_rate=attention_probs_dropout_prob,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_swish_activation=use_swish_activation
        )
        
        # Embedding layers
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # Dynamic aggregation for truth inference
        self.dynamic_aggregation = DynamicAggregation(
            hidden_size=hidden_size,
            num_labels=num_labels,
            num_contributors=num_contributors
        )
        
        # Final classifier
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Initialize with pretrained weights if provided
        if pretrained_model_path:
            self.load_pretrained(pretrained_model_path)
    
    def load_pretrained(self, pretrained_model_path):
        """Load pretrained model weights."""
        checkpoint = torch.load(pretrained_model_path, map_location='cpu')
              
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            self.load_state_dict(checkpoint, strict=False)
        
        print(f"Loaded pretrained model from {pretrained_model_path}")
    
    def get_input_embeddings(self, input_ids, token_type_ids=None, position_ids=None):
        """Get embeddings for input tokens."""
        seq_length = input_ids.size(1)
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Create token type IDs 
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Get embeddings
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # Combine embeddings
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        contributor_ids=None,
        contributor_labels=None,
        labels=None,
        return_dict=True
    ):
       
        #Forward pass of the TIA model.
        
        # Get input embeddings
        embeddings = self.get_input_embeddings(input_ids, token_type_ids, position_ids)
        
        # Reshape embeddings for Swin Transformer
        batch_size, seq_length, hidden_size = embeddings.shape
               
        side_length = int(seq_length**0.5) + 1
        padded_length = side_length * side_length
        
        if seq_length < padded_length:
            padding = torch.zeros(batch_size, padded_length - seq_length, hidden_size, device=embeddings.device)
            embeddings = torch.cat([embeddings, padding], dim=1)
        
        embeddings = embeddings.reshape(batch_size, side_length, side_length, hidden_size)
        embeddings = embeddings.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Pass through Swin Transformer
        contextualized_embeddings = self.swin_transformer(embeddings)
        contextualized_embeddings = contextualized_embeddings.reshape(batch_size, hidden_size)
        logits = self.classifier(contextualized_embeddings)
        
        # Dynamic truth inference 
        if contributor_ids is not None and contributor_labels is not None:
            inferred_labels, contributor_reliability = self.dynamic_aggregation(
                contextualized_embeddings, 
                contributor_ids, 
                contributor_labels
            )
        else:
            inferred_labels = torch.argmax(logits, dim=1)
            contributor_reliability = None
        
        # Loss calculation
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            # Reliability regularization if contributor information is available
            if contributor_reliability is not None:
                reliability_loss = self.dynamic_aggregation.reliability_loss(
                    contributor_reliability, 
                    inferred_labels, 
                    labels
                )
                loss += 0.1 * reliability_loss
        
        if return_dict:
            return {
                'logits': logits,
                'inferred_labels': inferred_labels,
                'contributor_reliability': contributor_reliability,
                'loss': loss
            }
        else:
            return (logits, inferred_labels, contributor_reliability, loss)

    def infer_truth(self, task_embeddings, contributor_labels):
        
        return self.dynamic_aggregation.infer(task_embeddings, contributor_labels)

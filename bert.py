from typing import Dict, List, Optional, Union, Tuple, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from utils import *


class BertSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # initialize the linear transformation layers for key, value, query
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # this dropout is applied to normalized attention scores following the original implementation of transformer
    # although it is a bit unusual, we empirically observe that it yields better performance
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # the corresponding linear_layer of k, v, q are used to project the hidden_state (x)
    bs, seq_len = x.shape[:2]
    proj = linear_layer(x)
    # next, we need to produce multiple heads for the proj 
    # this is done by spliting the hidden state to self.num_attention_heads, each of size self.attention_head_size
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    # by proper transpose, we have proj of [bs, num_attention_heads, seq_len, attention_head_size]
    proj = proj.transpose(1, 2)
    return proj

  def attention(self, key, query, value, attention_mask):
    # I calculate attention scores by taking the scaled dot-product between query and key.
    # This step allows me to determine the relevance of each token to every other token in the sequence.
    attention_scores = torch.matmul(query, key.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # I scale the scores to stabilize gradients.

    # I apply an attention mask to prevent the model from attending to padding tokens.
    # This ensures that the model focuses only on the meaningful parts of the input.
    attention_scores = attention_scores + attention_mask

    # I normalize the attention scores to probabilities using softmax.
    # This allows me to weigh the value vectors based on their relevance.
    attention_probs = F.softmax(attention_scores, dim=-1)
    attention_probs = self.dropout(attention_probs)  # I apply dropout to prevent overfitting.

    # I compute a weighted sum of value vectors based on the attention probabilities.
    # This creates a context-aware representation for each token.
    context_layer = torch.matmul(attention_probs, value)

    # I concatenate the attention heads and recover the original tensor shape.
    # This allows me to combine information from different representation subspaces.
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)
    return context_layer




  def forward(self, hidden_states, attention_mask):
    # I apply multi-head attention to the input hidden states.
    # This allows the model to attend to different parts of the sequence simultaneously.
    attention_output = self.self_attention(hidden_states, attention_mask)

    # I use an add-norm step to combine and normalize the output of the attention layer.
    attention_output = self.add_norm(hidden_states, attention_output, self.attention_dense, self.attention_dropout, self.attention_layer_norm)

    # I pass the attention output through a feed-forward network to introduce additional non-linearity.
    interm_output = self.interm_dense(attention_output)
    interm_output = self.interm_af(interm_output)

    # I apply another add-norm step to the output of the feed-forward network.
    # This ensures that the final output is well-normalized and stable.
    layer_output = self.add_norm(attention_output, interm_output, self.out_dense, self.out_dropout, self.out_layer_norm)
    return layer_output



class BertLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # multi-head attention
    self.self_attention = BertSelfAttention(config)
    # add-norm
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # feed forward
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # another add-norm
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add_norm(self, input, output, dense_layer, dropout, ln_layer):
    # I transform the output using a dense layer to project it into a different representation space.
    # This transformation allows the model to learn more complex relationships and patterns in the data.
    output = dense_layer(output)

    # I apply dropout to the transformed output to prevent overfitting.
    # Dropout randomly zeroes out some elements of the tensor, forcing the model to learn robust features that generalize well to unseen data.
    output = dropout(output)

    # I add the original input to the transformed output (residual connection) and normalize the result.
    # The residual connection helps with gradient flow and training stability, allowing the model to effectively learn deep representations.
    # Layer normalization stabilizes the input distribution, ensuring that the inputs to each layer are normalized and have a consistent scale.
    # This helps the model to train faster and more effectively.
    output = ln_layer(input + output)
    return output


def forward(self, hidden_states, attention_mask):
    # Multi-head attention
    attention_output = self.self_attention(hidden_states, attention_mask)

    # Add-norm
    attention_output = self.add_norm(hidden_states, attention_output, self.attention_dense, self.attention_dropout, self.attention_layer_norm)

    # Feed forward
    interm_output = self.interm_dense(attention_output)
    interm_output = self.interm_af(interm_output)

    # Another add-norm
    layer_output = self.add_norm(attention_output, interm_output, self.out_dense, self.out_dropout, self.out_layer_norm)
    return layer_output




class BertModel(BertPreTrainedModel):
  """
  the bert model returns the final embeddings for each token in a sentence
  it consists
  1. embedding (used in self.embed)
  2. a stack of n bert layers (used in self.encode)
  3. a linear transformation layer for [CLS] token (used in self.forward, as given)
  """
  def __init__(self, config):
    super().__init__(config)
    self.config = config

    # embedding
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
    self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
    # position_ids (1, len position emb) is a constant, register to buffer
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    # bert encoder
    self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    # for [CLS] token
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()

    self.init_weights()

  def embed(self, input_ids):
    input_shape = input_ids.size()
    seq_length = input_shape[1]

    # I look up word embeddings for the input token IDs.
    # These embeddings provide a dense representation of the tokens.
    input_embeds = self.word_embedding(input_ids)

    # I add positional embeddings to the word embeddings to encode the position of each token in the sequence.
    pos_ids = self.position_ids[:, :seq_length]
    pos_embeds = self.pos_embedding(pos_ids)

    # I use token type embeddings to provide additional information about the role of each token.
    # However, in this implementation, I'm using a placeholder as token types are not considered.
    tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
    tk_type_embeds = self.tk_type_embedding(tk_type_ids)

    # I sum the word, position, and token type embeddings to create a combined representation for each token.
    embeddings = input_embeds + pos_embeds + tk_type_embeds

    # I apply layer normalization and dropout to the combined embeddings to ensure they are well-regularized and stable.
    embeddings = self.embed_layer_norm(embeddings)
    embeddings = self.embed_dropout(embeddings)
    return embeddings




  def encode(self, hidden_states, attention_mask):
    """
    hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
    # get the extended attention mask for self attention
    # returns extended_attention_mask of [batch_size, 1, 1, seq_len]
    # non-padding tokens with 0 and padding tokens with a large negative number 
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

    # pass the hidden states through the encoder layers
    for i, layer_module in enumerate(self.bert_layers):
      # feed the encoding from the last bert_layer to the next
      hidden_states = layer_module(hidden_states, extended_attention_mask)

    return hidden_states

  def forward(self, input_ids, attention_mask):
    """
    input_ids: [batch_size, seq_len], seq_len is the max length of the batch
    attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
    """
    # get the embedding for each input token
    embedding_output = self.embed(input_ids=input_ids)

    # feed to a transformer (a stack of BertLayers)
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

    # get cls token hidden state
    first_tk = sequence_output[:, 0]
    first_tk = self.pooler_dense(first_tk)
    first_tk = self.pooler_af(first_tk)

    return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}

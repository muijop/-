// Neural network layers for AI training framework
// Provides common neural network layers similar to PyTorch

#ifndef NN_H
#define NN_H

#include "tensor.h"

// Linear (fully connected) layer
typedef struct {
    Tensor* weight;              // Weight matrix
    Tensor* bias;                 // Bias vector (optional)
    bool training;               // Training mode flag
} Linear;

// LSTM layer for sequence modeling
typedef struct {
    size_t input_size;            // Input feature size
    size_t hidden_size;           // Hidden state size
    size_t num_layers;            // Number of LSTM layers
    bool bidirectional;           // Bidirectional flag
    Linear** layers;              // LSTM layer parameters
} LSTM;

// Multi-head attention layer for transformers
typedef struct {
    size_t embed_dim;             // Embedding dimension
    size_t num_heads;             // Number of attention heads
    size_t head_dim;              // Dimension per head
    Linear* q_proj;               // Query projection
    Linear* k_proj;               // Key projection
    Linear* v_proj;               // Value projection
    Linear* out_proj;             // Output projection
} MultiheadAttention;

// Transformer encoder layer
typedef struct {
    size_t d_model;               // Model dimension
    size_t num_heads;             // Number of attention heads
    size_t d_ff;                  // Feed-forward dimension
    float dropout;                // Dropout rate
    MultiheadAttention* attention; // Attention layer
    Linear* linear1;              // First feed-forward linear layer
    Linear* linear2;              // Second feed-forward linear layer
    Tensor* norm1_weight;         // First normalization weight
    Tensor* norm1_bias;           // First normalization bias
    Tensor* norm2_weight;         // Second normalization weight
    Tensor* norm2_bias;           // Second normalization bias
} TransformerEncoderLayer;

// Transformer encoder stack
typedef struct {
    size_t d_model;               // Model dimension
    size_t num_heads;             // Number of attention heads
    size_t num_layers;            // Number of encoder layers
    size_t d_ff;                  // Feed-forward dimension
    float dropout;                // Dropout rate
    TransformerEncoderLayer** layers; // Encoder layers
    Tensor* norm_weight;          // Final normalization weight
    Tensor* norm_bias;            // Final normalization bias
} TransformerEncoder;

// Embedding layer for token embeddings
typedef struct {
    size_t vocab_size;            // Vocabulary size
    size_t d_model;               // Embedding dimension
    Tensor* weight;               // Embedding weight matrix
} Embedding;

// Layer normalization layer
typedef struct {
    size_t num_features;          // Number of features to normalize
    Tensor* weight;               // Scaling weights
    Tensor* bias;                 // Bias terms
    float eps;                    // Epsilon for numerical stability
} LayerNorm;

// Linear layer functions
Linear* linear_create(size_t in_features, size_t out_features, bool bias);
void linear_free(Linear* linear);
Tensor* linear_forward(Linear* linear, Tensor* input);

// LSTM layer functions
LSTM* lstm_create(size_t input_size, size_t hidden_size, size_t num_layers, bool bidirectional);
void lstm_free(LSTM* lstm);
Tensor* lstm_forward(LSTM* lstm, Tensor* input);

// Multi-head attention functions
MultiheadAttention* multihead_attention_create(size_t embed_dim, size_t num_heads);
void multihead_attention_free(MultiheadAttention* attention);
Tensor* multihead_attention_forward(MultiheadAttention* attention, Tensor* query, Tensor* key, Tensor* value);

// Transformer encoder layer functions
TransformerEncoderLayer* transformer_encoder_layer_create(size_t d_model, size_t num_heads, size_t d_ff, float dropout);
void transformer_encoder_layer_free(TransformerEncoderLayer* layer);
Tensor* transformer_encoder_layer_forward(TransformerEncoderLayer* layer, Tensor* src);

// Transformer encoder functions
TransformerEncoder* transformer_encoder_create(size_t d_model, size_t num_heads, size_t num_layers, size_t d_ff, float dropout);
void transformer_encoder_free(TransformerEncoder* encoder);
Tensor* transformer_encoder_forward(TransformerEncoder* encoder, Tensor* src);

// Embedding layer functions
Embedding* embedding_create(size_t vocab_size, size_t d_model);
void embedding_free(Embedding* embedding);
Tensor* embedding_forward(Embedding* embedding, Tensor* input);

// Layer normalization functions
LayerNorm* layer_norm_create(size_t num_features, float eps);
void layer_norm_free(LayerNorm* norm);
Tensor* layer_norm_forward(LayerNorm* norm, Tensor* input);

#endif
#include "nn.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

Linear* linear_create(size_t in_features, size_t out_features, bool bias) {
    Linear* linear = (Linear*)malloc(sizeof(Linear));
    if (!linear) return NULL;
    
    size_t weight_shape[2] = {out_features, in_features};
    linear->weight = tensor_randn(weight_shape, 2, true);
    
    if (bias) {
        size_t bias_shape[1] = {out_features};
        linear->bias = tensor_zeros(bias_shape, 1, true);
    } else {
        linear->bias = NULL;
    }
    
    linear->training = true;
    
    if (!linear->weight) {
        free(linear);
        return NULL;
    }
    
    float scale = sqrt(2.0f / in_features);
    for (size_t i = 0; i < linear->weight->size; i++) {
        linear->weight->data[i] *= scale;
    }
    
    return linear;
}

void linear_free(Linear* linear) {
    if (linear) {
        tensor_free(linear->weight);
        tensor_free(linear->bias);
        free(linear);
    }
}

Tensor* linear_forward(Linear* linear, Tensor* input) {
    if (input->ndim != 2) return NULL;
    if (input->shape[1] != linear->weight->shape[1]) return NULL;
    
    size_t out_shape[2] = {input->shape[0], linear->weight->shape[0]};
    Tensor* output = tensor_zeros(out_shape, 2, input->requires_grad || linear->weight->requires_grad);
    
    if (!output) return NULL;
    
    for (size_t i = 0; i < input->shape[0]; i++) {
        for (size_t j = 0; j < linear->weight->shape[0]; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < input->shape[1]; k++) {
                sum += input->data[i * input->strides[0] + k * input->strides[1]] *
                       linear->weight->data[j * linear->weight->strides[0] + k * linear->weight->strides[1]];
            }
            if (linear->bias) {
                sum += linear->bias->data[j];
            }
            output->data[i * output->strides[0] + j * output->strides[1]] = sum;
        }
    }
    
    return output;
}

MultiheadAttention* multihead_attention_create(size_t embed_dim, size_t num_heads) {
    MultiheadAttention* attention = (MultiheadAttention*)malloc(sizeof(MultiheadAttention));
    if (!attention) return NULL;
    
    attention->embed_dim = embed_dim;
    attention->num_heads = num_heads;
    attention->head_dim = embed_dim / num_heads;
    
    attention->q_proj = linear_create(embed_dim, embed_dim, false);
    attention->k_proj = linear_create(embed_dim, embed_dim, false);
    attention->v_proj = linear_create(embed_dim, embed_dim, false);
    attention->out_proj = linear_create(embed_dim, embed_dim, false);
    
    if (!attention->q_proj || !attention->k_proj || !attention->v_proj || !attention->out_proj) {
        multihead_attention_free(attention);
        return NULL;
    }
    
    return attention;
}

void multihead_attention_free(MultiheadAttention* attention) {
    if (attention) {
        linear_free(attention->q_proj);
        linear_free(attention->k_proj);
        linear_free(attention->v_proj);
        linear_free(attention->out_proj);
        free(attention);
    }
}

static Tensor* attention_scores(Tensor* query, Tensor* key) {
    size_t batch_size = query->shape[0];
    size_t seq_len = query->shape[1];
    size_t head_dim = query->shape[2] / query->shape[0];
    
    size_t scores_shape[3] = {batch_size, seq_len, seq_len};
    Tensor* scores = tensor_zeros(scores_shape, 3, query->requires_grad || key->requires_grad);
    
    if (!scores) return NULL;
    
    float scale = 1.0f / sqrt((float)head_dim);
    
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t j = 0; j < seq_len; j++) {
                float sum = 0.0f;
                for (size_t d = 0; d < head_dim; d++) {
                    sum += query->data[b * query->strides[0] + i * query->strides[1] + d * query->strides[2]] *
                           key->data[b * key->strides[0] + j * key->strides[1] + d * key->strides[2]];
                }
                scores->data[b * scores->strides[0] + i * scores->strides[1] + j * scores->strides[2]] = sum * scale;
            }
        }
    }
    
    return scores;
}

static Tensor* attention_weights(Tensor* scores) {
    return tensor_softmax(scores, 2);
}

static Tensor* attention_output(Tensor* weights, Tensor* value) {
    size_t batch_size = weights->shape[0];
    size_t seq_len = weights->shape[1];
    size_t head_dim = value->shape[2] / value->shape[0];
    
    size_t output_shape[3] = {batch_size, seq_len, head_dim};
    Tensor* output = tensor_zeros(output_shape, 3, weights->requires_grad || value->requires_grad);
    
    if (!output) return NULL;
    
    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < seq_len; i++) {
            for (size_t d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (size_t j = 0; j < seq_len; j++) {
                    sum += weights->data[b * weights->strides[0] + i * weights->strides[1] + j * weights->strides[2]] *
                           value->data[b * value->strides[0] + j * value->strides[1] + d * value->strides[2]];
                }
                output->data[b * output->strides[0] + i * output->strides[1] + d * output->strides[2]] = sum;
            }
        }
    }
    
    return output;
}

Tensor* multihead_attention_forward(MultiheadAttention* attention, Tensor* query, Tensor* key, Tensor* value) {
    if (query->ndim != 3 || key->ndim != 3 || value->ndim != 3) return NULL;
    if (query->shape[2] != attention->embed_dim) return NULL;
    
    Tensor* q = linear_forward(attention->q_proj, query);
    Tensor* k = linear_forward(attention->k_proj, key);
    Tensor* v = linear_forward(attention->v_proj, value);
    
    if (!q || !k || !v) {
        tensor_free(q);
        tensor_free(k);
        tensor_free(v);
        return NULL;
    }
    
    Tensor* scores = attention_scores(q, k);
    if (!scores) {
        tensor_free(q);
        tensor_free(k);
        tensor_free(v);
        return NULL;
    }
    
    Tensor* weights = attention_weights(scores);
    Tensor* output = attention_output(weights, v);
    
    if (!weights || !output) {
        tensor_free(q);
        tensor_free(k);
        tensor_free(v);
        tensor_free(scores);
        tensor_free(weights);
        tensor_free(output);
        return NULL;
    }
    
    Tensor* final_output = linear_forward(attention->out_proj, output);
    
    tensor_free(q);
    tensor_free(k);
    tensor_free(v);
    tensor_free(scores);
    tensor_free(weights);
    tensor_free(output);
    
    return final_output;
}

LayerNorm* layer_norm_create(size_t num_features, float eps) {
    LayerNorm* norm = (LayerNorm*)malloc(sizeof(LayerNorm));
    if (!norm) return NULL;
    
    norm->num_features = num_features;
    norm->eps = eps;
    
    size_t weight_shape[1] = {num_features};
    norm->weight = tensor_ones(weight_shape, 1, true);
    norm->bias = tensor_zeros(weight_shape, 1, true);
    
    if (!norm->weight || !norm->bias) {
        layer_norm_free(norm);
        return NULL;
    }
    
    return norm;
}

void layer_norm_free(LayerNorm* norm) {
    if (norm) {
        tensor_free(norm->weight);
        tensor_free(norm->bias);
        free(norm);
    }
}

Tensor* layer_norm_forward(LayerNorm* norm, Tensor* input) {
    if (input->shape[input->ndim - 1] != norm->num_features) return NULL;
    
    Tensor* result = tensor_create(NULL, input->shape, input->ndim, input->requires_grad);
    if (!result) return NULL;
    
    size_t batch_size = 1;
    for (size_t i = 0; i < input->ndim - 1; i++) {
        batch_size *= input->shape[i];
    }
    
    for (size_t b = 0; b < batch_size; b++) {
        float mean = 0.0f;
        for (size_t i = 0; i < norm->num_features; i++) {
            mean += input->data[b * norm->num_features + i];
        }
        mean /= norm->num_features;
        
        float var = 0.0f;
        for (size_t i = 0; i < norm->num_features; i++) {
            float diff = input->data[b * norm->num_features + i] - mean;
            var += diff * diff;
        }
        var /= norm->num_features;
        
        float std = sqrt(var + norm->eps);
        
        for (size_t i = 0; i < norm->num_features; i++) {
            size_t idx = b * norm->num_features + i;
            result->data[idx] = (input->data[idx] - mean) / std;
            result->data[idx] = result->data[idx] * norm->weight->data[i] + norm->bias->data[i];
        }
    }
    
    return result;
}

TransformerEncoderLayer* transformer_encoder_layer_create(size_t d_model, size_t num_heads, size_t d_ff, float dropout) {
    TransformerEncoderLayer* layer = (TransformerEncoderLayer*)malloc(sizeof(TransformerEncoderLayer));
    if (!layer) return NULL;
    
    layer->d_model = d_model;
    layer->num_heads = num_heads;
    layer->d_ff = d_ff;
    layer->dropout = dropout;
    
    layer->attention = multihead_attention_create(d_model, num_heads);
    layer->linear1 = linear_create(d_model, d_ff, true);
    layer->linear2 = linear_create(d_ff, d_model, true);
    
    size_t norm_shape[1] = {d_model};
    layer->norm1_weight = tensor_ones(norm_shape, 1, true);
    layer->norm1_bias = tensor_zeros(norm_shape, 1, true);
    layer->norm2_weight = tensor_ones(norm_shape, 1, true);
    layer->norm2_bias = tensor_zeros(norm_shape, 1, true);
    
    if (!layer->attention || !layer->linear1 || !layer->linear2 ||
        !layer->norm1_weight || !layer->norm1_bias || 
        !layer->norm2_weight || !layer->norm2_bias) {
        transformer_encoder_layer_free(layer);
        return NULL;
    }
    
    return layer;
}

void transformer_encoder_layer_free(TransformerEncoderLayer* layer) {
    if (layer) {
        multihead_attention_free(layer->attention);
        linear_free(layer->linear1);
        linear_free(layer->linear2);
        tensor_free(layer->norm1_weight);
        tensor_free(layer->norm1_bias);
        tensor_free(layer->norm2_weight);
        tensor_free(layer->norm2_bias);
        free(layer);
    }
}

Tensor* transformer_encoder_layer_forward(TransformerEncoderLayer* layer, Tensor* src) {
    Tensor* attn_output = multihead_attention_forward(layer->attention, src, src, src);
    if (!attn_output) return NULL;
    
    Tensor* src2 = tensor_add(src, attn_output);
    tensor_free(attn_output);
    if (!src2) return NULL;
    
    Tensor* norm1_output = layer_norm_forward((LayerNorm*)layer, src2);
    tensor_free(src2);
    if (!norm1_output) return NULL;
    
    Tensor* ff_output = linear_forward(layer->linear1, norm1_output);
    if (!ff_output) {
        tensor_free(norm1_output);
        return NULL;
    }
    
    Tensor* ff_output2 = linear_forward(layer->linear2, ff_output);
    tensor_free(ff_output);
    if (!ff_output2) {
        tensor_free(norm1_output);
        return NULL;
    }
    
    Tensor* src3 = tensor_add(norm1_output, ff_output2);
    tensor_free(norm1_output);
    tensor_free(ff_output2);
    if (!src3) return NULL;
    
    Tensor* output = layer_norm_forward((LayerNorm*)layer, src3);
    tensor_free(src3);
    
    return output;
}

TransformerEncoder* transformer_encoder_create(size_t d_model, size_t num_heads, size_t num_layers, size_t d_ff, float dropout) {
    TransformerEncoder* encoder = (TransformerEncoder*)malloc(sizeof(TransformerEncoder));
    if (!encoder) return NULL;
    
    encoder->d_model = d_model;
    encoder->num_heads = num_heads;
    encoder->num_layers = num_layers;
    encoder->d_ff = d_ff;
    encoder->dropout = dropout;
    
    encoder->layers = (TransformerEncoderLayer**)malloc(num_layers * sizeof(TransformerEncoderLayer*));
    if (!encoder->layers) {
        free(encoder);
        return NULL;
    }
    
    for (size_t i = 0; i < num_layers; i++) {
        encoder->layers[i] = transformer_encoder_layer_create(d_model, num_heads, d_ff, dropout);
        if (!encoder->layers[i]) {
            for (size_t j = 0; j < i; j++) {
                transformer_encoder_layer_free(encoder->layers[j]);
            }
            free(encoder->layers);
            free(encoder);
            return NULL;
        }
    }
    
    size_t norm_shape[1] = {d_model};
    encoder->norm_weight = tensor_ones(norm_shape, 1, true);
    encoder->norm_bias = tensor_zeros(norm_shape, 1, true);
    
    if (!encoder->norm_weight || !encoder->norm_bias) {
        transformer_encoder_free(encoder);
        return NULL;
    }
    
    return encoder;
}

void transformer_encoder_free(TransformerEncoder* encoder) {
    if (encoder) {
        for (size_t i = 0; i < encoder->num_layers; i++) {
            transformer_encoder_layer_free(encoder->layers[i]);
        }
        free(encoder->layers);
        tensor_free(encoder->norm_weight);
        tensor_free(encoder->norm_bias);
        free(encoder);
    }
}

Tensor* transformer_encoder_forward(TransformerEncoder* encoder, Tensor* src) {
    Tensor* output = src;
    
    for (size_t i = 0; i < encoder->num_layers; i++) {
        Tensor* new_output = transformer_encoder_layer_forward(encoder->layers[i], output);
        if (!new_output) return NULL;
        if (i > 0) tensor_free(output);
        output = new_output;
    }
    
    return output;
}
#include "reinforcement_learning.h"
#include "nn_module.h"
#include "tensor.h"
#include "autograd.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

// ===========================================
// 内部工具函数
// ===========================================

// 生成随机数（0-1）
static float random_float(void) {
    return (float)rand() / RAND_MAX;
}

// 生成高斯随机数
static float gaussian_random(float mean, float stddev) {
    float u1 = random_float();
    float u2 = random_float();
    
    while (u1 <= 0.0f) u1 = random_float();
    while (u2 <= 0.0f) u2 = random_float();
    
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    return mean + stddev * z0;
}

// 获取当前时间（秒）
static double get_current_time(void) {
    return (double)clock() / CLOCKS_PER_SEC;
}

// 计算Softmax
static void softmax(float* input, float* output, int size) {
    if (!input || !output || size <= 0) return;
    
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

// 裁剪值到范围
static float clip_value(float value, float min_val, float max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

// ===========================================
// 经验回放缓冲区实现
// ===========================================

experience_replay_t* create_experience_replay(int capacity, bool prioritized) {
    experience_replay_t* replay = 
        (experience_replay_t*)calloc(1, sizeof(experience_replay_t));
    
    if (!replay) {
        return NULL;
    }
    
    replay->capacity = capacity;
    replay->size = 0;
    replay->position = 0;
    replay->prioritized = prioritized;
    replay->alpha = 0.6f;
    replay->beta = 0.4f;
    replay->beta_increment = 0.001f;
    
    // 分配内存
    replay->states = (tensor_t**)calloc(capacity, sizeof(tensor_t*));
    replay->next_states = (tensor_t**)calloc(capacity, sizeof(tensor_t*));
    replay->actions = (tensor_t**)calloc(capacity, sizeof(tensor_t*));
    replay->rewards = (float*)calloc(capacity, sizeof(float));
    replay->dones = (bool*)calloc(capacity, sizeof(bool));
    
    if (!replay->states || !replay->next_states || !replay->actions || 
        !replay->rewards || !replay->dones) {
        destroy_experience_replay(replay);
        return NULL;
    }
    
    if (prioritized) {
        replay->priorities = (float*)calloc(capacity, sizeof(float));
        if (!replay->priorities) {
            destroy_experience_replay(replay);
            return NULL;
        }
    }
    
    return replay;
}

void destroy_experience_replay(experience_replay_t* replay) {
    if (!replay) return;
    
    if (replay->states) {
        for (int i = 0; i < replay->size; i++) {
            if (replay->states[i]) tensor_free(replay->states[i]);
        }
        free(replay->states);
    }
    
    if (replay->next_states) {
        for (int i = 0; i < replay->size; i++) {
            if (replay->next_states[i]) tensor_free(replay->next_states[i]);
        }
        free(replay->next_states);
    }
    
    if (replay->actions) {
        for (int i = 0; i < replay->size; i++) {
            if (replay->actions[i]) tensor_free(replay->actions[i]);
        }
        free(replay->actions);
    }
    
    if (replay->rewards) free(replay->rewards);
    if (replay->dones) free(replay->dones);
    if (replay->priorities) free(replay->priorities);
    
    free(replay);
}

int add_experience(experience_replay_t* replay, const tensor_t* state,
                  const tensor_t* action, float reward,
                  const tensor_t* next_state, bool done) {
    if (!replay || !state || !action || !next_state) {
        return -1;
    }
    
    // 复制张量
    tensor_t* state_copy = tensor_copy(state);
    tensor_t* action_copy = tensor_copy(action);
    tensor_t* next_state_copy = tensor_copy(next_state);
    
    if (!state_copy || !action_copy || !next_state_copy) {
        if (state_copy) tensor_free(state_copy);
        if (action_copy) tensor_free(action_copy);
        if (next_state_copy) tensor_free(next_state_copy);
        return -1;
    }
    
    // 替换或添加经验
    int idx = replay->position;
    
    // 释放旧经验
    if (replay->states[idx]) tensor_free(replay->states[idx]);
    if (replay->actions[idx]) tensor_free(replay->actions[idx]);
    if (replay->next_states[idx]) tensor_free(replay->next_states[idx]);
    
    // 存储新经验
    replay->states[idx] = state_copy;
    replay->actions[idx] = action_copy;
    replay->next_states[idx] = next_state_copy;
    replay->rewards[idx] = reward;
    replay->dones[idx] = done;
    
    // 设置优先级
    if (replay->prioritized) {
        replay->priorities[idx] = 1.0f;  // 初始优先级
    }
    
    replay->position = (replay->position + 1) % replay->capacity;
    if (replay->size < replay->capacity) {
        replay->size++;
    }
    
    return 0;
}

int sample_experience_batch(experience_replay_t* replay, int batch_size,
                          tensor_t** states, tensor_t** actions,
                          float* rewards, tensor_t** next_states,
                          bool* dones, float* weights, int* indices) {
    if (!replay || replay->size == 0 || batch_size <= 0) {
        return -1;
    }
    
    if (batch_size > replay->size) {
        batch_size = replay->size;
    }
    
    // 更新beta
    if (replay->prioritized) {
        replay->beta = fminf(1.0f, replay->beta + replay->beta_increment);
    }
    
    for (int i = 0; i < batch_size; i++) {
        int idx;
        
        if (replay->prioritized) {
            // 优先级采样
            float total_priority = 0.0f;
            for (int j = 0; j < replay->size; j++) {
                total_priority += powf(replay->priorities[j], replay->alpha);
            }
            
            float rand_val = random_float() * total_priority;
            float cumulative = 0.0f;
            
            for (idx = 0; idx < replay->size; idx++) {
                cumulative += powf(replay->priorities[idx], replay->alpha);
                if (cumulative >= rand_val) break;
            }
            
            if (idx >= replay->size) idx = replay->size - 1;
            
            // 计算重要性采样权重
            float prob = powf(replay->priorities[idx], replay->alpha) / total_priority;
            weights[i] = powf(replay->size * prob, -replay->beta);
        } else {
            // 均匀采样
            idx = rand() % replay->size;
            if (weights) weights[i] = 1.0f;
        }
        
        if (indices) indices[i] = idx;
        
        // 复制经验
        states[i] = tensor_copy(replay->states[idx]);
        actions[i] = tensor_copy(replay->actions[idx]);
        next_states[i] = tensor_copy(replay->next_states[idx]);
        rewards[i] = replay->rewards[idx];
        dones[i] = replay->dones[idx];
    }
    
    return batch_size;
}

void update_priorities(experience_replay_t* replay, const int* indices,
                      const float* priorities) {
    if (!replay || !replay->prioritized || !indices || !priorities) {
        return;
    }
    
    // 更新优先级
    for (int i = 0; indices[i] >= 0 && i < replay->size; i++) {
        int idx = indices[i];
        if (idx >= 0 && idx < replay->size) {
            replay->priorities[idx] = priorities[i];
        }
    }
}

// ===========================================
// 强化学习智能体实现
// ===========================================

reinforcement_learning_agent_t* create_reinforcement_learning_agent(
    reinforcement_learning_type_t type,
    const reinforcement_learning_config_t* config) {
    
    reinforcement_learning_agent_t* agent = 
        (reinforcement_learning_agent_t*)calloc(1, sizeof(reinforcement_learning_agent_t));
    
    if (!agent) {
        return NULL;
    }
    
    agent->type = type;
    
    if (config) {
        agent->config = *config;
    } else {
        // 设置默认配置
        switch (type) {
            case RL_DQN:
                agent->config = create_default_dqn_config();
                break;
            case RL_PPO:
                agent->config = create_default_ppo_config();
                break;
            case RL_A2C:
                agent->config = create_default_a2c_config();
                break;
            case RL_SAC:
                agent->config = create_default_sac_config();
                break;
            default:
                agent->config.method = type;
                agent->config.max_episodes = 1000;
                agent->config.max_steps_per_episode = 1000;
                agent->config.learning_rate = 0.001f;
                break;
        }
    }
    
    // 初始化随机种子
    srand((unsigned int)time(NULL));
    
    agent->current_episode = 0;
    agent->total_steps = 0;
    agent->total_reward = 0.0f;
    agent->best_reward = -FLT_MAX;
    agent->is_training = false;
    agent->is_initialized = false;
    
    printf("创建强化学习智能体，类型: %d\n", type);
    
    return agent;
}

void destroy_reinforcement_learning_agent(reinforcement_learning_agent_t* agent) {
    if (!agent) return;
    
    // 释放神经网络
    if (agent->policy_network) nn_module_free(agent->policy_network);
    if (agent->value_network) nn_module_free(agent->value_network);
    if (agent->target_network) nn_module_free(agent->target_network);
    if (agent->q_network1) nn_module_free(agent->q_network1);
    if (agent->q_network2) nn_module_free(agent->q_network2);
    if (agent->target_q_network1) nn_module_free(agent->target_q_network1);
    if (agent->target_q_network2) nn_module_free(agent->target_q_network2);
    
    // 释放经验回放
    if (agent->replay_buffer) destroy_experience_replay(agent->replay_buffer);
    
    // 释放环境
    if (agent->environment) destroy_environment(agent->environment);
    
    free(agent);
}

// ===========================================
// 配置设置
// ===========================================

int set_reinforcement_learning_config(reinforcement_learning_agent_t* agent,
                                   const reinforcement_learning_config_t* config) {
    if (!agent || !config) {
        return -1;
    }
    
    agent->config = *config;
    return 0;
}

int configure_reinforcement_learning_agent(reinforcement_learning_agent_t* agent,
                                          int state_dim, int action_dim, bool is_discrete) {
    if (!agent) {
        return -1;
    }
    
    // 创建神经网络
    switch (agent->type) {
        case RL_DQN: {
            // DQN网络：状态 -> Q值
            agent->policy_network = create_nn_module();
            
            // 输入层
            add_linear_layer(agent->policy_network, state_dim, 128);
            add_relu_layer(agent->policy_network);
            
            // 隐藏层
            add_linear_layer(agent->policy_network, 128, 64);
            add_relu_layer(agent->policy_network);
            
            // 输出层
            add_linear_layer(agent->policy_network, 64, action_dim);
            
            // 目标网络
            agent->target_network = nn_module_clone(agent->policy_network);
            
            // 经验回放
            agent->replay_buffer = create_experience_replay(
                agent->config.config.dqn.replay_capacity,
                agent->config.config.dqn.prioritized_replay);
            
            break;
        }
        
        case RL_PPO: {
            // PPO网络：状态 -> 策略（均值/标准差）和价值
            agent->policy_network = create_nn_module();
            
            // 共享特征提取
            add_linear_layer(agent->policy_network, state_dim, 64);
            add_tanh_layer(agent->policy_network);
            
            // 策略头
            add_linear_layer(agent->policy_network, 64, 64);
            add_tanh_layer(agent->policy_network);
            add_linear_layer(agent->policy_network, 64, action_dim * 2); // 均值和log_std
            
            // 价值网络
            agent->value_network = create_nn_module();
            add_linear_layer(agent->value_network, state_dim, 64);
            add_tanh_layer(agent->value_network);
            add_linear_layer(agent->value_network, 64, 64);
            add_tanh_layer(agent->value_network);
            add_linear_layer(agent->value_network, 64, 1);
            
            break;
        }
        
        case RL_SAC: {
            // SAC网络：状态 -> Q值（两个网络）和策略
            agent->q_network1 = create_nn_module();
            add_linear_layer(agent->q_network1, state_dim + action_dim, 256);
            add_relu_layer(agent->q_network1);
            add_linear_layer(agent->q_network1, 256, 256);
            add_relu_layer(agent->q_network1);
            add_linear_layer(agent->q_network1, 256, 1);
            
            agent->q_network2 = nn_module_clone(agent->q_network1);
            
            agent->target_q_network1 = nn_module_clone(agent->q_network1);
            agent->target_q_network2 = nn_module_clone(agent->q_network2);
            
            // 策略网络
            agent->policy_network = create_nn_module();
            add_linear_layer(agent->policy_network, state_dim, 256);
            add_relu_layer(agent->policy_network);
            add_linear_layer(agent->policy_network, 256, 256);
            add_relu_layer(agent->policy_network);
            add_linear_layer(agent->policy_network, 256, action_dim * 2);
            
            // 经验回放
            agent->replay_buffer = create_experience_replay(1000000, false);
            
            break;
        }
        
        default:
            printf("暂不支持的强化学习方法: %d\n", agent->type);
            return -1;
    }
    
    agent->is_initialized = true;
    printf("强化学习智能体配置完成，状态维度: %d，动作维度: %d\n", state_dim, action_dim);
    
    return 0;
}

// ===========================================
// 环境设置
// ===========================================

int set_reinforcement_learning_environment(reinforcement_learning_agent_t* agent,
                                          rl_environment_t* environment) {
    if (!agent || !environment) {
        return -1;
    }
    
    agent->environment = environment;
    
    // 配置智能体
    return configure_reinforcement_learning_agent(agent, 
                                                 environment->state_dim,
                                                 environment->action_dim,
                                                 environment->is_discrete);
}

rl_environment_t* create_simple_environment(int state_dim, int action_dim, bool is_discrete) {
    rl_environment_t* env = (rl_environment_t*)calloc(1, sizeof(rl_environment_t));
    
    if (!env) {
        return NULL;
    }
    
    env->name = strdup("Simple Environment");
    env->state_dim = state_dim;
    env->action_dim = action_dim;
    env->is_discrete = is_discrete;
    env->action_low = -1.0f;
    env->action_high = 1.0f;
    env->max_steps = 1000;
    
    return env;
}

void destroy_environment(rl_environment_t* environment) {
    if (!environment) return;
    
    free(environment->name);
    
    if (environment->close && environment->env_data) {
        environment->close(environment->env_data);
    }
    
    free(environment);
}

// ===========================================
// DQN算法实现
// ===========================================

static tensor_t* dqn_get_action(reinforcement_learning_agent_t* agent, const tensor_t* state, float epsilon) {
    if (!agent || !state || agent->type != RL_DQN) {
        return NULL;
    }
    
    // ε-贪婪策略
    if (random_float() < epsilon) {
        // 随机动作
        tensor_t* random_action = tensor_create_1d(agent->environment->action_dim);
        if (random_action) {
            if (agent->environment->is_discrete) {
                // 离散动作：选择随机动作
                int action = rand() % agent->environment->action_dim;
                for (int i = 0; i < agent->environment->action_dim; i++) {
                    random_action->data[i] = (i == action) ? 1.0f : 0.0f;
                }
            } else {
                // 连续动作：均匀随机
                for (int i = 0; i < agent->environment->action_dim; i++) {
                    random_action->data[i] = agent->environment->action_low + 
                                           random_float() * (agent->environment->action_high - agent->environment->action_low);
                }
            }
        }
        return random_action;
    } else {
        // 贪婪动作
        tensor_t* q_values = nn_module_forward(agent->policy_network, state);
        if (!q_values) return NULL;
        
        tensor_t* action = tensor_create_1d(agent->environment->action_dim);
        if (!action) {
            tensor_free(q_values);
            return NULL;
        }
        
        if (agent->environment->is_discrete) {
            // 离散动作：选择最大Q值的动作
            int best_action = 0;
            float max_q = q_values->data[0];
            
            for (int i = 1; i < agent->environment->action_dim; i++) {
                if (q_values->data[i] > max_q) {
                    max_q = q_values->data[i];
                    best_action = i;
                }
            }
            
            for (int i = 0; i < agent->environment->action_dim; i++) {
                action->data[i] = (i == best_action) ? 1.0f : 0.0f;
            }
        } else {
            // 连续动作：直接输出Q值（简化）
            for (int i = 0; i < agent->environment->action_dim; i++) {
                action->data[i] = clip_value(q_values->data[i], 
                                           agent->environment->action_low,
                                           agent->environment->action_high);
            }
        }
        
        tensor_free(q_values);
        return action;
    }
}

static int dqn_train_step(reinforcement_learning_agent_t* agent) {
    if (!agent || agent->type != RL_DQN || !agent->replay_buffer || 
        agent->replay_buffer->size < agent->config.config.dqn.batch_size) {
        return -1;
    }
    
    // 采样批次
    int batch_size = agent->config.config.dqn.batch_size;
    tensor_t** states = (tensor_t**)calloc(batch_size, sizeof(tensor_t*));
    tensor_t** actions = (tensor_t**)calloc(batch_size, sizeof(tensor_t*));
    float* rewards = (float*)calloc(batch_size, sizeof(float));
    tensor_t** next_states = (tensor_t**)calloc(batch_size, sizeof(tensor_t*));
    bool* dones = (bool*)calloc(batch_size, sizeof(bool));
    float* weights = (float*)calloc(batch_size, sizeof(float));
    int* indices = (int*)calloc(batch_size, sizeof(int));
    
    if (!states || !actions || !rewards || !next_states || !dones) {
        if (states) free(states);
        if (actions) free(actions);
        if (rewards) free(rewards);
        if (next_states) free(next_states);
        if (dones) free(dones);
        if (weights) free(weights);
        if (indices) free(indices);
        return -1;
    }
    
    int actual_batch_size = sample_experience_batch(agent->replay_buffer, batch_size,
                                                  states, actions, rewards, next_states, dones,
                                                  weights, indices);
    
    if (actual_batch_size <= 0) {
        free(states); free(actions); free(rewards); free(next_states); free(dones);
        free(weights); free(indices);
        return -1;
    }
    
    // 计算目标Q值
    float* target_q = (float*)calloc(actual_batch_size, sizeof(float));
    float* priorities = (float*)calloc(actual_batch_size, sizeof(float));
    
    if (!target_q || !priorities) {
        free(states); free(actions); free(rewards); free(next_states); free(dones);
        free(weights); free(indices); free(target_q); free(priorities);
        return -1;
    }
    
    for (int i = 0; i < actual_batch_size; i++) {
        if (dones[i]) {
            target_q[i] = rewards[i];
        } else {
            // 使用目标网络计算下一状态的Q值
            tensor_t* next_q_values = nn_module_forward(agent->target_network, next_states[i]);
            if (!next_q_values) {
                target_q[i] = rewards[i];
            } else {
                float max_next_q = next_q_values->data[0];
                for (int j = 1; j < agent->environment->action_dim; j++) {
                    if (next_q_values->data[j] > max_next_q) {
                        max_next_q = next_q_values->data[j];
                    }
                }
                target_q[i] = rewards[i] + agent->config.config.dqn.gamma * max_next_q;
                tensor_free(next_q_values);
            }
        }
        
        // 计算TD误差（用于优先级）
        tensor_t* current_q_values = nn_module_forward(agent->policy_network, states[i]);
        if (current_q_values) {
            float current_q = 0.0f;
            if (agent->environment->is_discrete) {
                // 找到动作对应的Q值
                for (int j = 0; j < agent->environment->action_dim; j++) {
                    if (actions[i]->data[j] > 0.5f) {
                        current_q = current_q_values->data[j];
                        break;
                    }
                }
            } else {
                // 连续动作：使用平均Q值（简化）
                for (int j = 0; j < agent->environment->action_dim; j++) {
                    current_q += current_q_values->data[j];
                }
                current_q /= agent->environment->action_dim;
            }
            
            float td_error = fabsf(target_q[i] - current_q);
            priorities[i] = td_error + 1e-6f;  // 避免零优先级
            
            tensor_free(current_q_values);
        }
    }
    
    // 更新优先级
    if (agent->config.config.dqn.prioritized_replay) {
        update_priorities(agent->replay_buffer, indices, priorities);
    }
    
    // 简化训练：随机更新网络参数
    for (int i = 0; i < agent->policy_network->num_layers; i++) {
        nn_layer_t* layer = agent->policy_network->layers[i];
        if (layer->weights) {
            for (int j = 0; j < layer->weights->shape[0] * layer->weights->shape[1]; j++) {
                layer->weights->data[j] -= agent->config.learning_rate * (random_float() - 0.5f) * 0.1f;
            }
        }
        if (layer->bias) {
            for (int j = 0; j < layer->bias->shape[0]; j++) {
                layer->bias->data[j] -= agent->config.learning_rate * (random_float() - 0.5f) * 0.1f;
            }
        }
    }
    
    // 更新目标网络
    if (agent->total_steps % agent->config.config.dqn.target_update_freq == 0) {
        nn_module_free(agent->target_network);
        agent->target_network = nn_module_clone(agent->policy_network);
    }
    
    // 清理
    for (int i = 0; i < actual_batch_size; i++) {
        if (states[i]) tensor_free(states[i]);
        if (actions[i]) tensor_free(actions[i]);
        if (next_states[i]) tensor_free(next_states[i]);
    }
    
    free(states); free(actions); free(rewards); free(next_states); free(dones);
    free(weights); free(indices); free(target_q); free(priorities);
    
    return 0;
}

// ===========================================
// 训练控制
// ===========================================

int start_reinforcement_learning_training(reinforcement_learning_agent_t* agent) {
    if (!agent || !agent->is_initialized || !agent->environment) {
        return -1;
    }
    
    if (agent->is_training) {
        printf("错误：强化学习已在训练中\n");
        return -1;
    }
    
    printf("开始强化学习训练，方法: %d，最大回合: %d\n", 
           agent->type, agent->config.max_episodes);
    
    agent->is_training = true;
    agent->current_episode = 0;
    agent->total_steps = 0;
    agent->total_reward = 0.0f;
    agent->best_reward = -FLT_MAX;
    
    double start_time = get_current_time();
    
    // 训练循环
    for (int episode = 0; episode < agent->config.max_episodes && agent->is_training; episode++) {
        agent->current_episode = episode;
        
        // 重置环境
        tensor_t* state = agent->environment->reset(agent->environment->env_data);
        if (!state) {
            printf("错误：环境重置失败\n");
            break;
        }
        
        float episode_reward = 0.0f;
        int episode_steps = 0;
        bool done = false;
        
        // 回合循环
        while (!done && episode_steps < agent->environment->max_steps && agent->is_training) {
            // 选择动作
            float epsilon = 0.0f;
            if (agent->type == RL_DQN) {
                epsilon = agent->config.config.dqn.epsilon_end + 
                         (agent->config.config.dqn.epsilon_start - agent->config.config.dqn.epsilon_end) *
                         expf(-1.0f * agent->total_steps / agent->config.config.dqn.epsilon_decay);
            }
            
            tensor_t* action = get_action_with_exploration(agent, state, epsilon);
            if (!action) {
                printf("错误：动作选择失败\n");
                break;
            }
            
            // 执行动作
            float reward;
            tensor_t* next_state = agent->environment->step(agent->environment->env_data, 
                                                         action, &reward, &done);
            
            if (!next_state) {
                printf("错误：环境步进失败\n");
                tensor_free(action);
                break;
            }
            
            // 存储经验
            if (agent->replay_buffer) {
                add_experience(agent->replay_buffer, state, action, reward, next_state, done);
            }
            
            episode_reward += reward;
            episode_steps++;
            agent->total_steps++;
            
            // 训练步骤
            if (agent->total_steps % agent->config.update_frequency == 0) {
                switch (agent->type) {
                    case RL_DQN:
                        dqn_train_step(agent);
                        break;
                    default:
                        // 其他方法暂不实现
                        break;
                }
            }
            
            // 更新状态
            tensor_free(state);
            state = next_state;
            tensor_free(action);
            
            // 检查时间限制
            if (agent->config.max_time_seconds > 0) {
                double elapsed = get_current_time() - start_time;
                if (elapsed > agent->config.max_time_seconds) {
                    printf("达到时间限制，停止训练\n");
                    agent->is_training = false;
                    break;
                }
            }
        }
        
        agent->total_reward += episode_reward;
        
        if (episode_reward > agent->best_reward) {
            agent->best_reward = episode_reward;
        }
        
        printf("回合 %d: 奖励=%.2f, 步数=%d, 总步数=%d\n", 
               episode, episode_reward, episode_steps, agent->total_steps);
        
        // 调用回调函数
        if (agent->progress_callback) {
            agent->progress_callback(episode, episode_reward, episode_steps);
        }
        
        if (agent->episode_complete_callback) {
            agent->episode_complete_callback(episode, episode_reward, episode_steps);
        }
        
        // 检查早停条件
        if (agent->config.early_stopping && episode_reward >= agent->config.target_reward) {
            printf("达到目标奖励，停止训练\n");
            agent->is_training = false;
            break;
        }
        
        // 清理状态
        if (state) tensor_free(state);
    }
    
    agent->is_training = false;
    
    double total_time = get_current_time() - start_time;
    printf("强化学习训练完成，总时间: %.2f秒，最佳奖励: %.2f\n", total_time, agent->best_reward);
    
    // 调用训练完成回调
    if (agent->training_complete_callback) {
        agent->training_complete_callback(agent->best_reward, agent->current_episode);
    }
    
    return 0;
}

int stop_reinforcement_learning_training(reinforcement_learning_agent_t* agent) {
    if (!agent) {
        return -1;
    }
    
    if (agent->is_training) {
        agent->is_training = false;
        printf("强化学习训练已停止\n");
        return 0;
    }
    
    return -1;
}

int pause_reinforcement_learning_training(reinforcement_learning_agent_t* agent) {
    if (!agent) {
        return -1;
    }
    
    if (agent->is_training) {
        agent->is_training = false;
        printf("强化学习训练已暂停\n");
        return 0;
    }
    
    return -1;
}

int resume_reinforcement_learning_training(reinforcement_learning_agent_t* agent) {
    if (!agent) {
        return -1;
    }
    
    if (!agent->is_training && agent->current_episode < agent->config.max_episodes) {
        agent->is_training = true;
        printf("强化学习训练已恢复\n");
        return 0;
    }
    
    return -1;
}

// ===========================================
// 推理和决策
// ===========================================

tensor_t* get_action(reinforcement_learning_agent_t* agent, const tensor_t* state) {
    return get_action_with_exploration(agent, state, 0.0f);
}

tensor_t* get_action_with_exploration(reinforcement_learning_agent_t* agent, 
                                     const tensor_t* state, float epsilon) {
    if (!agent || !state || !agent->is_initialized) {
        return NULL;
    }
    
    switch (agent->type) {
        case RL_DQN:
            return dqn_get_action(agent, state, epsilon);
        default:
            printf("暂不支持的强化学习方法: %d\n", agent->type);
            return NULL;
    }
}

// ===========================================
// 结果获取
// ===========================================

reinforcement_learning_result_t* get_reinforcement_learning_result(
    const reinforcement_learning_agent_t* agent) {
    if (!agent) {
        return NULL;
    }
    
    // 创建结果结构体（简化实现）
    reinforcement_learning_result_t* result = 
        (reinforcement_learning_result_t*)calloc(1, sizeof(reinforcement_learning_result_t));
    
    if (!result) {
        return NULL;
    }
    
    result->final_reward = agent->best_reward;
    result->best_reward = agent->best_reward;
    result->total_steps = agent->total_steps;
    result->success = agent->best_reward >= agent->config.target_reward;
    result->status_message = strdup("训练完成");
    
    return result;
}

float get_current_reward(const reinforcement_learning_agent_t* agent) {
    if (!agent) {
        return 0.0f;
    }
    
    return agent->best_reward;
}

int get_current_episode(const reinforcement_learning_agent_t* agent) {
    if (!agent) {
        return 0;
    }
    
    return agent->current_episode;
}

// ===========================================
// 模型保存和加载
// ===========================================

int save_reinforcement_learning_model(const reinforcement_learning_agent_t* agent,
                                    const char* filepath) {
    if (!agent || !filepath) {
        return -1;
    }
    
    printf("保存强化学习模型到: %s\n", filepath);
    
    // 简化实现：只保存网络结构信息
    FILE* file = fopen(filepath, "w");
    if (!file) {
        return -1;
    }
    
    fprintf(file, "Reinforcement Learning Model\n");
    fprintf(file, "Type: %d\n", agent->type);
    fprintf(file, "State Dim: %d\n", agent->environment ? agent->environment->state_dim : 0);
    fprintf(file, "Action Dim: %d\n", agent->environment ? agent->environment->action_dim : 0);
    fprintf(file, "Best Reward: %.4f\n", agent->best_reward);
    
    fclose(file);
    
    return 0;
}

int load_reinforcement_learning_model(reinforcement_learning_agent_t* agent,
                                    const char* filepath) {
    if (!agent || !filepath) {
        return -1;
    }
    
    printf("从文件加载强化学习模型: %s\n", filepath);
    
    // 简化实现：只读取基本信息
    FILE* file = fopen(filepath, "r");
    if (!file) {
        return -1;
    }
    
    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (strstr(line, "Best Reward:")) {
            float best_reward;
            if (sscanf(line, "Best Reward: %f", &best_reward) == 1) {
                agent->best_reward = best_reward;
            }
        }
    }
    
    fclose(file);
    
    return 0;
}

// ===========================================
// 回调函数设置
// ===========================================

void set_reinforcement_learning_progress_callback(reinforcement_learning_agent_t* agent,
                                                 void (*callback)(int episode, float reward, int steps)) {
    if (agent) {
        agent->progress_callback = callback;
    }
}

void set_episode_complete_callback(reinforcement_learning_agent_t* agent,
                                 void (*callback)(int episode, float reward, int steps)) {
    if (agent) {
        agent->episode_complete_callback = callback;
    }
}

void set_training_complete_callback(reinforcement_learning_agent_t* agent,
                                  void (*callback)(float final_reward, int total_episodes)) {
    if (agent) {
        agent->training_complete_callback = callback;
    }
}

// ===========================================
// 工具函数
// ===========================================

reinforcement_learning_config_t create_default_dqn_config(void) {
    reinforcement_learning_config_t config;
    
    config.method = RL_DQN;
    config.max_episodes = 1000;
    config.max_steps_per_episode = 1000;
    config.learning_rate = 0.001f;
    config.update_frequency = 4;
    config.use_gpu = false;
    config.seed = 42;
    
    config.config.dqn.gamma = 0.99f;
    config.config.dqn.epsilon_start = 1.0f;
    config.config.dqn.epsilon_end = 0.01f;
    config.config.dqn.epsilon_decay = 10000;
    config.config.dqn.target_update_freq = 100;
    config.config.dqn.batch_size = 32;
    config.config.dqn.replay_capacity = 10000;
    config.config.dqn.double_dqn = true;
    config.config.dqn.dueling_dqn = true;
    config.config.dqn.prioritized_replay = true;
    
    config.hidden_layers = 2;
    config.hidden_units = 64;
    config.activation = "relu";
    
    config.early_stopping = true;
    config.target_reward = 195.0f;  // CartPole目标奖励
    config.patience = 10;
    
    config.max_time_seconds = 3600;
    config.max_memory_mb = 4096;
    
    return config;
}

reinforcement_learning_config_t create_default_ppo_config(void) {
    reinforcement_learning_config_t config;
    
    config.method = RL_PPO;
    config.max_episodes = 1000;
    config.max_steps_per_episode = 1000;
    config.learning_rate = 0.0003f;
    config.update_frequency = 1;
    config.use_gpu = false;
    config.seed = 42;
    
    config.config.ppo.gamma = 0.99f;
    config.config.ppo.lambda = 0.95f;
    config.config.ppo.clip_epsilon = 0.2f;
    config.config.ppo.epochs = 10;
    config.config.ppo.batch_size = 64;
    config.config.ppo.learning_rate = 0.0003f;
    config.config.ppo.entropy_coef = 0.01f;
    config.config.ppo.value_coef = 0.5f;
    config.config.ppo.horizon = 2048;
    config.config.ppo.use_gae = true;
    
    config.hidden_layers = 2;
    config.hidden_units = 64;
    config.activation = "tanh";
    
    config.early_stopping = true;
    config.target_reward = 200.0f;
    config.patience = 10;
    
    config.max_time_seconds = 3600;
    config.max_memory_mb = 4096;
    
    return config;
}

reinforcement_learning_config_t create_default_a2c_config(void) {
    reinforcement_learning_config_t config;
    
    config.method = RL_A2C;
    config.max_episodes = 1000;
    config.max_steps_per_episode = 1000;
    config.learning_rate = 0.0007f;
    config.update_frequency = 5;
    config.use_gpu = false;
    config.seed = 42;
    
    config.config.a2c.gamma = 0.99f;
    config.config.a2c.learning_rate = 0.0007f;
    config.config.a2c.entropy_coef = 0.01f;
    config.config.a2c.value_coef = 0.5f;
    config.config.a2c.n_steps = 5;
    config.config.a2c.use_gae = true;
    config.config.a2c.gae_lambda = 0.95f;
    
    config.hidden_layers = 2;
    config.hidden_units = 64;
    config.activation = "relu";
    
    config.early_stopping = true;
    config.target_reward = 195.0f;
    config.patience = 10;
    
    config.max_time_seconds = 3600;
    config.max_memory_mb = 4096;
    
    return config;
}

reinforcement_learning_config_t create_default_sac_config(void) {
    reinforcement_learning_config_t config;
    
    config.method = RL_SAC;
    config.max_episodes = 1000;
    config.max_steps_per_episode = 1000;
    config.learning_rate = 0.0003f;
    config.update_frequency = 1;
    config.use_gpu = false;
    config.seed = 42;
    
    config.config.sac.gamma = 0.99f;
    config.config.sac.tau = 0.005f;
    config.config.sac.alpha = 0.2f;
    config.config.sac.auto_entropy_tuning = true;
    config.config.sac.target_entropy = -1.0f;
    config.config.sac.learning_rate = 0.0003f;
    config.config.sac.batch_size = 256;
    
    config.hidden_layers = 2;
    config.hidden_units = 256;
    config.activation = "relu";
    
    config.early_stopping = true;
    config.target_reward = -250.0f;  // Pendulum目标奖励
    config.patience = 10;
    
    config.max_time_seconds = 3600;
    config.max_memory_mb = 4096;
    
    return config;
}

// ===========================================
// 环境工具函数
// ===========================================

rl_environment_t* create_cartpole_environment(void) {
    rl_environment_t* env = create_simple_environment(4, 2, true);
    if (env) {
        free(env->name);
        env->name = strdup("CartPole-v1");
        env->max_steps = 500;
    }
    return env;
}

rl_environment_t* create_mountain_car_environment(void) {
    rl_environment_t* env = create_simple_environment(2, 3, true);
    if (env) {
        free(env->name);
        env->name = strdup("MountainCar-v0");
        env->max_steps = 200;
    }
    return env;
}

rl_environment_t* create_pendulum_environment(void) {
    rl_environment_t* env = create_simple_environment(3, 1, false);
    if (env) {
        free(env->name);
        env->name = strdup("Pendulum-v1");
        env->action_low = -2.0f;
        env->action_high = 2.0f;
        env->max_steps = 200;
    }
    return env;
}

rl_environment_t* create_lunar_lander_environment(void) {
    rl_environment_t* env = create_simple_environment(8, 4, true);
    if (env) {
        free(env->name);
        env->name = strdup("LunarLander-v2");
        env->max_steps = 1000;
    }
    return env;
}
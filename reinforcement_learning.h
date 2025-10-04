#ifndef REINFORCEMENT_LEARNING_H
#define REINFORCEMENT_LEARNING_H

#include "nn_module.h"
#include "tensor.h"
#include <stdbool.h>

// ===========================================
// 强化学习类型枚举
// ===========================================

typedef enum {
    RL_DQN = 0,                    // Deep Q-Network
    RL_PPO = 1,                    // Proximal Policy Optimization
    RL_A2C = 2,                   // Advantage Actor-Critic
    RL_SAC = 3,                   // Soft Actor-Critic
    RL_TD3 = 4,                   // Twin Delayed DDPG
    RL_DDPG = 5,                  // Deep Deterministic Policy Gradient
    RL_REINFORCE = 6,             // REINFORCE (Policy Gradient)
    RL_A3C = 7                    // Asynchronous Advantage Actor-Critic
} reinforcement_learning_type_t;

// ===========================================
// 环境接口定义
// ===========================================

typedef struct {
    char* name;                   // 环境名称
    int state_dim;                // 状态维度
    int action_dim;               // 动作维度
    bool is_discrete;             // 是否为离散动作空间
    float action_low;             // 动作下限（连续动作）
    float action_high;            // 动作上限（连续动作）
    int max_steps;                // 最大步数
    
    // 环境重置函数
    tensor_t* (*reset)(void* env_data);
    // 环境步进函数
    tensor_t* (*step)(void* env_data, const tensor_t* action, float* reward, bool* done);
    // 环境渲染函数
    void (*render)(void* env_data);
    // 环境关闭函数
    void (*close)(void* env_data);
    
    void* env_data;               // 环境特定数据
} rl_environment_t;

// ===========================================
// 经验回放缓冲区
// ===========================================

typedef struct {
    tensor_t** states;            // 状态数组
    tensor_t** next_states;       // 下一状态数组
    tensor_t** actions;           // 动作数组
    float* rewards;               // 奖励数组
    bool* dones;                  // 终止标志数组
    int capacity;                 // 缓冲区容量
    int size;                     // 当前大小
    int position;                 // 当前位置
    bool prioritized;             // 是否使用优先级经验回放
    float* priorities;            // 优先级数组
    float alpha;                  // 优先级指数
    float beta;                   // 重要性采样权重
    float beta_increment;         // beta增量
} experience_replay_t;

// ===========================================
// DQN配置结构体
// ===========================================

typedef struct {
    float gamma;                  // 折扣因子
    float epsilon_start;          // 初始探索率
    float epsilon_end;            // 最终探索率
    float epsilon_decay;          // 探索率衰减
    int target_update_freq;       // 目标网络更新频率
    int batch_size;               // 批次大小
    int replay_capacity;          // 回放缓冲区容量
    bool double_dqn;              // 是否使用Double DQN
    bool dueling_dqn;             // 是否使用Dueling DQN
    bool prioritized_replay;      // 是否使用优先级经验回放
} dqn_config_t;

// ===========================================
// PPO配置结构体
// ===========================================

typedef struct {
    float gamma;                  // 折扣因子
    float lambda;                 // GAE lambda参数
    float clip_epsilon;           // PPO裁剪参数
    int epochs;                   // 每次更新的轮次
    int batch_size;               // 批次大小
    float learning_rate;          // 学习率
    float entropy_coef;           // 熵系数
    float value_coef;             // 价值函数系数
    int horizon;                  // 轨迹长度
    bool use_gae;                 // 是否使用GAE
} ppo_config_t;

// ===========================================
// A2C配置结构体
// ===========================================

typedef struct {
    float gamma;                  // 折扣因子
    float learning_rate;          // 学习率
    float entropy_coef;           // 熵系数
    float value_coef;             // 价值函数系数
    int n_steps;                  // n步回报
    bool use_gae;                 // 是否使用GAE
    float gae_lambda;             // GAE lambda参数
} a2c_config_t;

// ===========================================
// SAC配置结构体
// ===========================================

typedef struct {
    float gamma;                  // 折扣因子
    float tau;                    // 目标网络软更新参数
    float alpha;                  // 温度参数
    bool auto_entropy_tuning;     // 是否自动调整熵
    float target_entropy;         // 目标熵
    float learning_rate;          // 学习率
    int batch_size;               // 批次大小
} sac_config_t;

// ===========================================
// 强化学习配置结构体
// ===========================================

typedef struct {
    reinforcement_learning_type_t method;  // 强化学习方法
    
    // 通用配置
    int max_episodes;             // 最大训练回合数
    int max_steps_per_episode;    // 每回合最大步数
    float learning_rate;          // 学习率
    int update_frequency;         // 更新频率
    bool use_gpu;                 // 是否使用GPU
    int seed;                     // 随机种子
    
    // 方法特定配置
    union {
        dqn_config_t dqn;
        ppo_config_t ppo;
        a2c_config_t a2c;
        sac_config_t sac;
    } config;
    
    // 网络架构配置
    int hidden_layers;            // 隐藏层数量
    int hidden_units;            // 隐藏层单元数
    char* activation;            // 激活函数
    
    // 训练控制
    bool early_stopping;          // 是否早停
    float target_reward;          // 目标奖励
    int patience;                 // 早停耐心值
    
    // 资源限制
    int max_time_seconds;         // 最大训练时间（秒）
    int max_memory_mb;            // 最大内存使用（MB）
} reinforcement_learning_config_t;

// ===========================================
// 强化学习智能体结构体
// ===========================================

typedef struct {
    reinforcement_learning_type_t type;    // 智能体类型
    reinforcement_learning_config_t config; // 配置
    
    // 神经网络模型
    nn_module_t* policy_network;           // 策略网络
    nn_module_t* value_network;            // 价值网络（如果需要）
    nn_module_t* target_network;            // 目标网络（如果需要）
    nn_module_t* q_network1;               // Q网络1（SAC/TD3）
    nn_module_t* q_network2;               // Q网络2（SAC/TD3）
    nn_module_t* target_q_network1;        // 目标Q网络1
    nn_module_t* target_q_network2;        // 目标Q网络2
    
    // 经验回放
    experience_replay_t* replay_buffer;     // 经验回放缓冲区
    
    // 训练状态
    int current_episode;          // 当前回合
    int total_steps;              // 总步数
    float total_reward;           // 总奖励
    float best_reward;            // 最佳奖励
    bool is_training;             // 是否在训练中
    bool is_initialized;          // 是否已初始化
    
    // 环境
    rl_environment_t* environment; // 强化学习环境
    
    // 回调函数
    void (*progress_callback)(int episode, float reward, int steps);
    void (*episode_complete_callback)(int episode, float reward, int steps);
    void (*training_complete_callback)(float final_reward, int total_episodes);
} reinforcement_learning_agent_t;

// ===========================================
// 训练结果结构体
// ===========================================

typedef struct {
    float* episode_rewards;       // 每回合奖励
    int* episode_steps;           // 每回合步数
    float* episode_losses;       // 每回合损失
    int num_episodes;            // 回合数量
    float final_reward;          // 最终奖励
    float best_reward;           // 最佳奖励
    int total_steps;             // 总步数
    float total_training_time;    // 总训练时间
    bool success;                // 是否成功
    char* status_message;        // 状态消息
} reinforcement_learning_result_t;

// ===========================================
// API函数声明
// ===========================================

// 智能体管理
reinforcement_learning_agent_t* create_reinforcement_learning_agent(
    reinforcement_learning_type_t type,
    const reinforcement_learning_config_t* config);
void destroy_reinforcement_learning_agent(reinforcement_learning_agent_t* agent);

// 配置设置
int set_reinforcement_learning_config(reinforcement_learning_agent_t* agent,
                                   const reinforcement_learning_config_t* config);
int configure_reinforcement_learning_agent(reinforcement_learning_agent_t* agent,
                                          int state_dim, int action_dim, bool is_discrete);

// 环境设置
int set_reinforcement_learning_environment(reinforcement_learning_agent_t* agent,
                                          rl_environment_t* environment);
rl_environment_t* create_simple_environment(int state_dim, int action_dim, bool is_discrete);
void destroy_environment(rl_environment_t* environment);

// 训练控制
int start_reinforcement_learning_training(reinforcement_learning_agent_t* agent);
int stop_reinforcement_learning_training(reinforcement_learning_agent_t* agent);
int pause_reinforcement_learning_training(reinforcement_learning_agent_t* agent);
int resume_reinforcement_learning_training(reinforcement_learning_agent_t* agent);

// 推理和决策
tensor_t* get_action(reinforcement_learning_agent_t* agent, const tensor_t* state);
tensor_t* get_action_with_exploration(reinforcement_learning_agent_t* agent, 
                                     const tensor_t* state, float epsilon);

// 结果获取
reinforcement_learning_result_t* get_reinforcement_learning_result(
    const reinforcement_learning_agent_t* agent);
float get_current_reward(const reinforcement_learning_agent_t* agent);
int get_current_episode(const reinforcement_learning_agent_t* agent);

// 模型保存和加载
int save_reinforcement_learning_model(const reinforcement_learning_agent_t* agent,
                                    const char* filepath);
int load_reinforcement_learning_model(reinforcement_learning_agent_t* agent,
                                    const char* filepath);

// 回调函数设置
void set_reinforcement_learning_progress_callback(reinforcement_learning_agent_t* agent,
                                                 void (*callback)(int episode, float reward, int steps));
void set_episode_complete_callback(reinforcement_learning_agent_t* agent,
                                 void (*callback)(int episode, float reward, int steps));
void set_training_complete_callback(reinforcement_learning_agent_t* agent,
                                  void (*callback)(float final_reward, int total_episodes));

// 工具函数
reinforcement_learning_config_t create_default_dqn_config(void);
reinforcement_learning_config_t create_default_ppo_config(void);
reinforcement_learning_config_t create_default_a2c_config(void);
reinforcement_learning_config_t create_default_sac_config(void);

// 经验回放管理
experience_replay_t* create_experience_replay(int capacity, bool prioritized);
void destroy_experience_replay(experience_replay_t* replay);
int add_experience(experience_replay_t* replay, const tensor_t* state,
                  const tensor_t* action, float reward,
                  const tensor_t* next_state, bool done);
int sample_experience_batch(experience_replay_t* replay, int batch_size,
                          tensor_t** states, tensor_t** actions,
                          float* rewards, tensor_t** next_states,
                          bool* dones, float* weights, int* indices);
void update_priorities(experience_replay_t* replay, const int* indices,
                      const float* priorities);

// 环境工具函数
rl_environment_t* create_cartpole_environment(void);
l_environment_t* create_mountain_car_environment(void);
rl_environment_t* create_pendulum_environment(void);
rl_environment_t* create_lunar_lander_environment(void);

#endif // REINFORCEMENT_LEARNING_H
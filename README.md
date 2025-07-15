# 🚀 KVCache 命中率计算器

## 🌐 在线使用
**直接访问**: [https://KevinCheung2259.github.io/kvcache-hit-rate-calculator/](https://KevinCheung2259.github.io/kvcache-hit-rate-calculator/)

> 将上面链接中的"你的用户名"替换为您的GitHub用户名

## 📋 功能特性

用于计算LLM推理服务中KVCache的理论命中率。本工具基于排队论和缓存理论，提供准确的性能预测和优化建议。

- 🎯 **精确建模**: 基于排队论和缓存理论的数学建模
- 📊 **可视化界面**: 现代化Web界面，支持实时计算和图表展示
- 🔧 **参数化配置**: 支持模型层数、注意力头数、数据类型等多种参数
- 💡 **优化建议**: 自动分析并提供内存配置优化建议
- 📈 **敏感性分析**: 分析不同参数对命中率的影响
- 🎨 **预设配置**: 内置主流模型配置（Llama2、ChatGLM等）

## 🏗️ 项目结构

```
kvcache-hit-rate-calculator/
├── kvcache_calculator.py    # 核心计算逻辑
├── index.html              # Web界面
├── style.css               # 样式文件
├── calculator.js           # 前端JavaScript逻辑
├── example.py              # Python使用示例
├── test.py                 # 测试套件
├── README.md               # 项目文档
├── requirements.txt        # 依赖管理
└── LICENSE                 # 开源许可证
```

## 🚀 快速开始

### 方式1: Web界面 (推荐)

1. 直接打开 `index.html` 文件
2. 在浏览器中设置模型和系统参数
3. 点击"计算命中率"按钮查看结果

### 方式2: Python脚本

```bash
# 运行示例
python example.py

# 或者直接使用API
python -c "
from kvcache_calculator import *
calculator = KVCacheCalculator()
# ... 你的代码
"
```

## 📊 核心概念

### KVCache 工作原理

在LLM推理中，KVCache存储了注意力机制的Key和Value矩阵，避免重复计算：

```
Memory per token = 2 × num_layers × num_kv_heads × head_dim × dtype_bytes
```

### 命中率计算模型

本工具基于以下理论模型：

1. **Little's Law**: `平均会话数 = 到达率 × 平均停留时间`
2. **LRU缓存策略**: 最近最少使用的缓存替换算法
3. **会话级建模**: 考虑同一会话内的时间局部性

### 关键指标

- **命中率**: KVCache命中的请求比例
- **缓存利用率**: 缓存空间的使用效率
- **延迟减少**: 由于缓存命中带来的性能提升
- **内存效率**: 缓存内存的有效利用程度

## 🧮 数学公式

### 1. 内存计算

**KVCache每Token内存:**
```
memory_per_token = 2 × num_layers × num_kv_heads × head_dim × dtype_bytes
```

其中:
- `2` 代表Key和Value
- `num_layers`: 模型层数
- `num_kv_heads`: Key-Value头数
- `head_dim`: 每个注意力头的维度
- `dtype_bytes`: 每个元素的字节数 (FP16=2, FP8=1等)

**最大缓存Token数:**
```
max_cached_tokens = (available_memory - model_memory × 1.2) / memory_per_token
```

### 2. 会话动态

**活跃会话数 (Little's Law):**
```
active_conversations = conversation_arrival_rate × conversation_lifetime
conversation_lifetime = avg_conversation_length × within_conversation_interval
```

**最大可缓存会话数:**
```
max_cached_conversations = max_cached_tokens / avg_tokens_per_conversation
avg_tokens_per_conversation = avg_conversation_length × avg_sequence_length
```

### 3. 命中率计算

**情况1: 缓存充足 (active_conversations ≤ max_cached_conversations)**
```
hit_rate = 1 - (1 / avg_conversation_length)
```

**情况2: 缓存不足 (active_conversations > max_cached_conversations)**
```
cache_ratio = max_cached_conversations / active_conversations
intra_conversation_hit = 1 - (1 / avg_conversation_length)
inter_conversation_hit = cache_ratio
hit_rate = intra_conversation_hit × inter_conversation_hit
```

### 4. 推导指标

**推导QPS:**
```
qps_per_conversation = avg_sequence_length / within_conversation_interval
derived_qps = conversation_arrival_rate × qps_per_conversation
```

**性能指标:**
```
tokens_per_second = derived_qps × avg_sequence_length
cache_hits_per_second = tokens_per_second × hit_rate
estimated_latency_reduction = hit_rate × 0.3  # 假设减少30%
```

## 🔧 参数说明

### 模型配置

| 参数 | 说明 | 示例值 |
|------|------|--------|
| 模型层数 | Transformer层数 | 32 |
| 注意力头数 | 注意力头的数量 | 32 |
| Key-Value头数 | Key-Value头数量 (用于GQA) | 32 |
| 头维度 | 每个注意力头的维度 | 128 |
| 模型大小 | 模型大小(GB) | 14 |
| KVCache数据类型 | KVCache存储的数据类型 | FP16 |

### 系统配置

| 参数 | 说明 | 示例值 |
|------|------|--------|
| 可用内存 | GPU/系统可用内存(GB) | 80 |

### 会话模式

| 参数 | 说明 | 示例值 |
|------|------|--------|
| 平均会话长度 | 每个会话的平均轮次数 | 5 |
| 新会话到达率 | 新会话开始的频率(会话/秒) | 2 |
| 会话内间隔 | 同一会话中请求的时间间隔(秒) | 30 |
| 平均序列长度 | 每个请求的平均token数 | 100 |

## 📈 使用示例

### Python API示例

```python
from kvcache_calculator import *

# 创建配置
model_config = ModelConfig(
    num_layers=32,
    num_attention_heads=32,
    num_kv_heads=32,
    head_dim=128,
    model_dtype=ModelDtype.FP16,
    kvcache_dtype=KVCacheDtype.FP16,
    model_size_gb=14.0
)

system_config = SystemConfig(
    available_memory_gb=80.0
)

conv_pattern = ConversationPattern(
    avg_conversation_length=5.0,
    conversation_arrival_rate=2.0,
    within_conversation_interval=30.0,
    avg_sequence_length=100
)

# 计算指标
calculator = KVCacheCalculator()
metrics = calculator.calculate_detailed_metrics(
    model_config, system_config, conv_pattern
)

print(f"KVCache命中率: {metrics['hit_rate']:.1%}")
print(f"缓存利用率: {metrics['cache_utilization']:.1%}")
print(f"预估延迟减少: {metrics['estimated_latency_reduction']:.1%}")
```

### 预设配置

工具内置了常见模型的配置：

- **Llama2-7B**: 32层, 32头, FP16
- **Llama2-13B**: 40层, 40头, FP16  
- **ChatGLM-6B**: 28层, 32头, FP16
- **大型模型**: 80层, 64头, FP16

## 🔍 理论背景

### 数学建模

该工具基于以下数学模型：

1. **内存使用模型**:
   ```
   Cache_Memory = Cached_Tokens × Memory_per_Token
   Memory_per_Token = 2 × L × H × D × B
   ```
   其中: L=层数, H=KV头数, D=头维度, B=数据类型字节数

2. **命中率模型**:
   ```
   Hit_Rate = P(intra_conversation) × P(cache_available)
   P(intra_conversation) = 1 - 1/avg_conversation_length
   P(cache_available) = min(1, cache_capacity/active_conversations)
   ```

3. **系统建模** (基于Little's Law):
   ```
   Active_Conversations = Arrival_Rate × Conversation_Lifetime
   Conversation_Lifetime = Avg_Length × Within_Interval
   ```

### 假设条件

- 平均序列长度可配置（默认100个token）
- LRU缓存替换策略
- 模型运行时内存开销为模型大小的1.2倍
- 缓存命中可减少30%的计算时间

## 📊 性能分析

### 影响因素分析

1. **内存大小**: 更大内存 → 更多缓存 → 更高命中率
2. **会话模式**: 更长会话 → 更高命中率
3. **数据类型**: 低精度类型 → 更小内存占用 → 更多缓存
4. **系统负载**: 更高QPS → 更多竞争 → 可能降低命中率

### 优化建议

1. **内存优化**: 
   - 使用FP8、INT8或FP16精度的KVCache以减少内存占用
   - FP8提供了精度和内存效率的良好平衡
   - 根据业务需求选择合适的内存配置

2. **系统设计**:
   - 考虑会话亲和性的负载均衡
   - 优化会话分发策略

3. **模型选择**:
   - 在精度和内存效率间平衡
   - 考虑使用MQA/GQA减少KV头数

## 🧪 测试

```bash
# 运行所有测试
python test.py

# 运行示例
python example.py
```

## 📄 许可证

MIT License - 请查看 LICENSE 文件了解详情

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个工具！

### 开发环境

```bash
# 安装依赖
pip install numpy

# 运行测试
python example.py
```

### 扩展建议

- [ ] 支持更多缓存策略（LFU, FIFO等）
- [ ] 添加GPU内存碎片化建模
- [ ] 支持多实例并发分析
- [ ] 集成更多模型架构预设

## 🙏 致谢

感谢以下资源和项目的启发：
- Transformer架构论文
- 各种开源LLM项目
- 缓存理论和排队论相关研究

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

💻 基于排队论等理论建模计算 | 实际效果可能因实现细节而异 
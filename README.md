# 🚀 KVCache Hit Rate Calculator

## 🌐 Online Usage
**Direct Access**: [https://KevinCheung2259.github.io/kvcache-hit-rate-calculator/](https://KevinCheung2259.github.io/kvcache-hit-rate-calculator/)

## 📋 Features

A tool for calculating the theoretical KVCache hit rate in LLM inference services. Based on queuing theory and cache theory, it provides accurate performance predictions and optimization suggestions.

- 🎯 **Precise Modeling**: Mathematical modeling based on queuing theory and cache theory
- 📊 **Visual Interface**: Modern web interface with real-time calculation and chart display
- 🔧 **Parameterized Configuration**: Supports model layers, KV heads, data types, and other parameters
- 💡 **Optimization Suggestions**: Automatic analysis and memory configuration optimization recommendations
- 📈 **Sensitivity Analysis**: Analysis of how different parameters affect hit rates
- 🎨 **Preset Configurations**: Built-in mainstream model configurations (Mistral, Llama3, Qwen3, etc.)

## 🏗️ Project Structure

```
kvcache-hit-rate-calculator/
├── kvcache_calculator.py    # Core calculation logic
├── index.html              # Web interface
├── style.css               # Style files
├── calculator.js           # Frontend JavaScript logic
├── example.py              # Python usage examples
├── test.py                 # Test suite
├── README.md               # Project documentation
├── requirements.txt        # Dependency management
└── LICENSE                 # Open source license
```

## 🚀 Quick Start

### Method1: Web Interface (Recommended)

1. Open [https://KevinCheung2259.github.io/kvcache-hit-rate-calculator/](https://KevinCheung2259.github.io/kvcache-hit-rate-calculator/)
2. Fill in the model configuration, system configuration, and conversation pattern parameters
   - Model Configuration: Number of layers, KV heads, head dimension, data type
   - System Configuration: Available memory
   - Conversation Pattern: Average conversation length, new conversation arrival rate, within conversation interval, average sequence length

### Method2: Python Script

```bash
# Run examples
python example.py

# Or use API directly
python -c "
from kvcache_calculator import *
calculator = KVCacheCalculator()
# ... your code
```

## 📊 Core Concepts

### How KVCache Works

In LLM inference, KVCache stores the Key and Value matrices of the attention mechanism to avoid repeated calculations:

```
Memory per token = 2 × num_layers × num_kv_heads × head_dim × dtype_bytes
```

### Hit Rate Calculation Model

This tool is based on the following theoretical models:
1. **Littles Law**: `Average conversations = Arrival rate × Average stay time`
2. **LRU Cache Strategy**: Least Recently Used cache replacement algorithm
3. **Conversation-level Modeling**: Considering temporal locality within the same conversation

### Key Metrics

- **Hit Rate**: Proportion of requests that hit KVCache
- **Cache Utilization**: Efficiency of cache space usage
- **System QPS**: Queries per second the system can handle
- **Memory Efficiency**: Effective utilization of cache memory

## 🧮 Mathematical Formulas

### 1. Memory Calculation

**KVCache Memory per Token:**
```
memory_per_token = 2 × num_layers × num_kv_heads × head_dim × dtype_bytes
```

Where:
- `2` represents Key and Value
- `num_layers`: Number of model layers
- `num_kv_heads`: Number of Key-Value heads
- `head_dim`: Dimension of each attention head
- `dtype_bytes`: Bytes per element (FP16=2, FP8=1, etc.)

**Maximum Cached Tokens:**
```
max_cached_tokens = (available_memory - model_memory × 1.2) / memory_per_token
```

### 2. Conversation Dynamics

**Active Conversations (Little's Law):**
```
active_conversations = conversation_arrival_rate × conversation_lifetime
conversation_lifetime = avg_conversation_length × within_conversation_interval
```

**Maximum Cached Conversations:**
```
max_cached_conversations = max_cached_tokens / avg_tokens_per_conversation
avg_tokens_per_conversation = avg_conversation_length × avg_sequence_length
```

### 3. Hit Rate Calculation

**Case 1: Sufficient Cache (active_conversations ≤ max_cached_conversations)**
```
hit_rate = 1 - (1 / avg_conversation_length)
```

**Case 2: Insufficient Cache (active_conversations > max_cached_conversations)**
```
cache_ratio = max_cached_conversations / active_conversations
intra_conversation_hit = 1 - (1 / avg_conversation_length)
inter_conversation_hit = cache_ratio
hit_rate = intra_conversation_hit × inter_conversation_hit
```

### 4. Derived Metrics

**Derived QPS:**
```
qps_per_conversation = avg_sequence_length / within_conversation_interval
derived_qps = conversation_arrival_rate × qps_per_conversation
```

**Performance Metrics:**
```
tokens_per_second = derived_qps × avg_sequence_length
cache_hits_per_second = tokens_per_second × hit_rate
```


## 📊 Performance Analysis

### Factor Analysis
1. **Memory Size**: Larger memory → More cache → Higher hit rate
2. **Conversation Pattern**: Longer conversations → Higher hit rate
3. **Data Type**: Lower precision types → Smaller memory usage → More cache
4. **System Load**: Higher QPS → More competition → May reduce hit rate

### Optimization Suggestions

1. **Memory Optimization**: 
   - Use FP8, INT8, or FP16 precision KVCache to reduce memory usage
   - FP8 provides a good balance between precision and memory efficiency
   - Choose appropriate memory configuration based on business requirements

2. **System Design**:
   - Consider session affinity load balancing
   - Optimize conversation distribution strategies

3. **Model Selection**:
   - Balance between precision and memory efficiency
   - Consider using MQA/GQA to reduce KV heads

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Welcome to submit Issues and Pull Requests to improve this tool!

### Extension Suggestions

- [ ] Support more cache management policies (FIFO, LFU, etc.)
- [ ] Support multi-instance with different schedule strategies
- [ ] Integrate more model architecture presets

## 🙏 Acknowledgments

Thanks to the following resources and projects for inspiration:
- Transformer architecture papers
- Various open-source LLM projects
- Cache theory and queuing theory related research

## 📞 Contact

For questions or suggestions, please contact through:
- Submit GitHub Issues
- Send email to me

---

💻 Based on queuing theory and other theoretical modeling | Actual performance may vary due to implementation details 

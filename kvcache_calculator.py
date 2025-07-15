#!/usr/bin/env python3
"""
KVCache命中率计算器
专门用于建模和计算LLM推理服务中KVCache命中率的工具
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict

class ModelDtype(Enum):
    """模型数据类型"""
    FP16 = "fp16"
    FP32 = "fp32"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"

class KVCacheDtype(Enum):
    """KVCache数据类型"""
    FP16 = "fp16"
    FP32 = "fp32"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"

@dataclass
class ModelConfig:
    """模型配置参数"""
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int          # 对于GQA，等于num_attention_heads对于MHA
    head_dim: int
    model_dtype: ModelDtype
    kvcache_dtype: KVCacheDtype
    model_size_gb: float  # 模型大小（GB）

@dataclass  
class SystemConfig:
    """系统配置参数"""
    available_memory_gb: float  # 可用内存（GB）

@dataclass
class ConversationPattern:
    """会话模式参数"""
    avg_conversation_length: float      # 平均会话长度（轮次）
    conversation_arrival_rate: float    # 新会话到达率（会话/秒）
    within_conversation_interval: float # 会话内平均间隔（秒）
    avg_sequence_length: int            # 平均序列长度（tokens）

class KVCacheCalculator:
    """KVCache命中率计算器"""
    
    def __init__(self):
        self.dtype_bytes = {
            ModelDtype.FP32: 4,
            ModelDtype.FP16: 2,
            ModelDtype.BF16: 2,
            ModelDtype.FP8: 1,
            ModelDtype.INT8: 1,
            ModelDtype.INT4: 0.5,
            KVCacheDtype.FP32: 4,
            KVCacheDtype.FP16: 2,
            KVCacheDtype.BF16: 2,
            KVCacheDtype.FP8: 1,
            KVCacheDtype.INT8: 1,
        }
    
    def calculate_kvcache_memory_per_token(self, model_config: ModelConfig) -> int:
        """计算每个token的KVCache内存占用（字节）"""
        # KVCache包含Key和Value，每层都有
        # 内存 = 2 (K+V) * num_layers * num_kv_heads * head_dim * dtype_bytes
        bytes_per_element = self.dtype_bytes[model_config.kvcache_dtype]
        memory_per_token = (2 * model_config.num_layers * 
                           model_config.num_kv_heads * 
                           model_config.head_dim * 
                           bytes_per_element)
        return memory_per_token
    
    def calculate_derived_qps(self, conv_pattern: ConversationPattern) -> float:
        """从会话参数推导QPS"""
        # 每个会话的平均QPS = 每轮tokens / 会话内间隔
        qps_per_conversation = conv_pattern.avg_sequence_length / conv_pattern.within_conversation_interval
        # 系统总QPS = 会话到达率 * 每个会话的QPS
        derived_qps = conv_pattern.conversation_arrival_rate * qps_per_conversation
        return derived_qps
    
    def calculate_max_cached_tokens(self, model_config: ModelConfig, 
                                   system_config: SystemConfig) -> int:
        """计算最大可缓存的token数量"""
        memory_per_token = self.calculate_kvcache_memory_per_token(model_config)
        
        # 可用于KVCache的内存 = 总内存 - 模型内存 - 系统开销
        available_for_cache = (system_config.available_memory_gb * 1024**3 - 
                              model_config.model_size_gb * 1024**3 * 1.2)  # 1.2倍模型内存开销
        
        if available_for_cache <= 0:
            return 0
            
        max_tokens = int(available_for_cache / memory_per_token)
        return max_tokens
    
    def calculate_conversation_hit_rate(self, model_config: ModelConfig,
                                      system_config: SystemConfig,
                                      conv_pattern: ConversationPattern) -> Dict[str, float]:
        """计算会话级别的命中率"""
        
        max_cached_tokens = self.calculate_max_cached_tokens(model_config, system_config)
        
        if max_cached_tokens <= 0:
            return {"hit_rate": 0.0, "avg_cached_conversations": 0.0, "cache_utilization": 0.0, 
                   "max_cached_tokens": 0, "active_conversations": 0}
        
        # 估算平均每个会话的token数
        avg_tokens_per_conversation = conv_pattern.avg_conversation_length * conv_pattern.avg_sequence_length
        
        # 可以缓存的会话数量
        max_cached_conversations = max_cached_tokens / avg_tokens_per_conversation
        
        # 会话到达率和生存时间建模
        conversation_lifetime = conv_pattern.avg_conversation_length * conv_pattern.within_conversation_interval
        
        # 使用Little's Law: 系统中的平均会话数 = 到达率 × 平均停留时间
        active_conversations = conv_pattern.conversation_arrival_rate * conversation_lifetime
        
        # 缓存命中率建模（基于LRU策略）
        if active_conversations <= max_cached_conversations:
            # 所有活跃会话都能被缓存
            hit_rate = 1.0 - (1.0 / conv_pattern.avg_conversation_length)  # 首轮请求无法命中
        else:
            # 部分会话被缓存，使用概率模型
            cache_ratio = max_cached_conversations / active_conversations
            # 考虑会话内的命中率和会话间的竞争
            intra_conversation_hit = 1.0 - (1.0 / conv_pattern.avg_conversation_length)
            inter_conversation_hit = cache_ratio
            hit_rate = intra_conversation_hit * inter_conversation_hit
        
        cache_utilization = min(active_conversations / max_cached_conversations, 1.0) if max_cached_conversations > 0 else 0
        
        return {
            "hit_rate": max(0.0, min(1.0, hit_rate)),
            "avg_cached_conversations": min(active_conversations, max_cached_conversations),
            "cache_utilization": cache_utilization,
            "max_cached_tokens": max_cached_tokens,
            "active_conversations": active_conversations
        }
    
    def calculate_detailed_metrics(self, model_config: ModelConfig,
                                 system_config: SystemConfig,
                                 conv_pattern: ConversationPattern) -> Dict[str, float]:
        """计算详细的性能指标"""
        
        basic_metrics = self.calculate_conversation_hit_rate(model_config, system_config, conv_pattern)
        
        # 计算内存使用详情
        memory_per_token = self.calculate_kvcache_memory_per_token(model_config)
        model_memory_gb = model_config.model_size_gb * 1.2  # 包含运行时开销
        
        # 计算性能提升 - QPS从会话参数推导
        derived_qps = self.calculate_derived_qps(conv_pattern)
        tokens_per_second = derived_qps * conv_pattern.avg_sequence_length
        cache_hits_per_second = tokens_per_second * basic_metrics["hit_rate"]
        
        # 估算延迟改善（缓存命中可以减少计算时间）
        compute_reduction_factor = 0.3  # 假设缓存命中减少30%计算时间
        avg_latency_reduction = basic_metrics["hit_rate"] * compute_reduction_factor
        
        return {
            **basic_metrics,
            "memory_per_token_bytes": memory_per_token,
            "model_memory_gb": model_memory_gb,
            "cache_memory_gb": (basic_metrics["max_cached_tokens"] * memory_per_token) / (1024**3),
            "derived_qps": derived_qps,
            "tokens_per_second": tokens_per_second,
            "cache_hits_per_second": cache_hits_per_second,
            "estimated_latency_reduction": avg_latency_reduction,
            "memory_efficiency": basic_metrics["cache_utilization"]
        }

    def optimize_memory_allocation(self, model_config: ModelConfig,
                                 system_config: SystemConfig,
                                 conv_pattern: ConversationPattern,
                                 target_hit_rate: float = 0.8) -> Dict[str, float]:
        """优化内存分配以达到目标命中率"""
        
        current_metrics = self.calculate_detailed_metrics(model_config, system_config, conv_pattern)
        
        if current_metrics["hit_rate"] >= target_hit_rate:
            return {"recommended_memory_gb": system_config.available_memory_gb,
                   "current_hit_rate": current_metrics["hit_rate"],
                   "achievable": True}
        
        # 估算达到目标命中率所需的内存
        memory_per_token = self.calculate_kvcache_memory_per_token(model_config)
        avg_tokens_per_conversation = conv_pattern.avg_conversation_length * conv_pattern.avg_sequence_length
        conversation_lifetime = conv_pattern.avg_conversation_length * conv_pattern.within_conversation_interval
        active_conversations = conv_pattern.conversation_arrival_rate * conversation_lifetime
        
        # 为了达到目标命中率，需要缓存更多会话
        required_cached_conversations = active_conversations * target_hit_rate
        required_tokens = required_cached_conversations * avg_tokens_per_conversation
        required_cache_memory = required_tokens * memory_per_token / (1024**3)
        
        required_total_memory = required_cache_memory + model_config.model_size_gb * 1.2
        
        return {
            "recommended_memory_gb": required_total_memory,
            "current_hit_rate": current_metrics["hit_rate"],
            "target_hit_rate": target_hit_rate,
            "additional_memory_needed_gb": max(0, required_total_memory - system_config.available_memory_gb),
            "achievable": required_total_memory <= system_config.available_memory_gb * 2  # 假设最多2倍当前内存
        } 
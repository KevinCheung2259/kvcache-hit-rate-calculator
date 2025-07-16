#!/usr/bin/env python3
"""
KVCache Hit Rate Calculator
A tool specifically designed to model and calculate the hit rate of KVCache in LLM inference services
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict

class ModelDtype(Enum):
    """Model data type"""
    FP16 = "fp16"
    FP32 = "fp32"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"

class KVCacheDtype(Enum):
    """KVCache data type"""
    FP16 = "fp16"
    FP32 = "fp32"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"

@dataclass
class ModelConfig:
    """Model configuration parameters"""
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int          # For GQA, equal to num_attention_heads for MHA
    head_dim: int
    model_dtype: ModelDtype
    kvcache_dtype: KVCacheDtype
    model_size_gb: float  # Model size (GB)

@dataclass  
class SystemConfig:
    """System configuration parameters"""
    available_memory_gb: float  # Available memory (GB)

@dataclass
class ConversationPattern:
    """Conversation pattern parameters"""
    avg_conversation_length: float      # Average conversation length (rounds)
    conversation_arrival_rate: float    # New conversation arrival rate (conversations/second)
    within_conversation_interval: float # Average interval within a conversation (seconds)
    avg_sequence_length: int            # Average sequence length (tokens)

class KVCacheCalculator:
    """KVCache Hit Rate Calculator"""
    
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
        """Calculate the memory occupied by each token in KVCache (bytes)"""
        # KVCache contains Key and Value, each layer has
        # Memory = 2 (K+V) * num_layers * num_kv_heads * head_dim * dtype_bytes
        bytes_per_element = self.dtype_bytes[model_config.kvcache_dtype]
        memory_per_token = (2 * model_config.num_layers * 
                           model_config.num_kv_heads * 
                           model_config.head_dim * 
                           bytes_per_element)
        return memory_per_token
    
    def calculate_derived_qps(self, conv_pattern: ConversationPattern) -> float:
        """Derive QPS from conversation parameters"""
        # QPS = number of requests per second, independent of token length
        # Based on Little's Law:
        # System QPS = conversation arrival rate × average conversation length (number of requests per conversation)
        return conv_pattern.conversation_arrival_rate * conv_pattern.avg_conversation_length
    
    def calculate_max_cached_tokens(self, model_config: ModelConfig, 
                                   system_config: SystemConfig) -> int:
        """Calculate the maximum number of tokens that can be cached"""
        memory_per_token = self.calculate_kvcache_memory_per_token(model_config)
        
        # Available memory for KVCache = total memory - model memory - system overhead
        available_for_cache = (system_config.available_memory_gb * 1024**3 - 
                              model_config.model_size_gb * 1024**3 * 1.2)  # 1.2x model memory overhead
        
        if available_for_cache <= 0:
            return 0
            
        max_tokens = int(available_for_cache / memory_per_token)
        return max_tokens
    
    def calculate_conversation_hit_rate(self, model_config: ModelConfig,
                                      system_config: SystemConfig,
                                      conv_pattern: ConversationPattern) -> Dict[str, float]:
        """Calculate the hit rate at the conversation level"""
        
        max_cached_tokens = self.calculate_max_cached_tokens(model_config, system_config)
        
        if max_cached_tokens <= 0:
            return {"hit_rate": 0.0, "avg_cached_conversations": 0.0, "cache_utilization": 0.0, 
                   "max_cached_tokens": 0, "active_conversations": 0}
        
        # Estimate the average number of tokens per conversation
        avg_tokens_per_conversation = conv_pattern.avg_conversation_length * conv_pattern.avg_sequence_length
        
        # Number of conversations that can be cached
        max_cached_conversations = max_cached_tokens / avg_tokens_per_conversation
        
        # Conversation arrival rate and survival time modeling
        conversation_lifetime = conv_pattern.avg_conversation_length * conv_pattern.within_conversation_interval
        
        # Using Little's Law: average number of conversations in the system = arrival rate × average stay time
        active_conversations = conv_pattern.conversation_arrival_rate * conversation_lifetime
        
        # Cache hit rate modeling (based on LRU strategy)
        if active_conversations <= max_cached_conversations:
            # All active conversations can be cached
            hit_rate = 1.0 - (1.0 / conv_pattern.avg_conversation_length)  # First request cannot hit
        else:
            # Some conversations are cached, using probability model
            cache_ratio = max_cached_conversations / active_conversations
            # Consider the hit rate within conversations and the competition between conversations
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
        """Calculate detailed performance metrics"""
        
        basic_metrics = self.calculate_conversation_hit_rate(model_config, system_config, conv_pattern)
        
        # Calculate memory usage details
        memory_per_token = self.calculate_kvcache_memory_per_token(model_config)
        model_memory_gb = model_config.model_size_gb * 1.2  # Includes runtime overhead
        
        # Calculate performance improvement - QPS derived from conversation parameters
        derived_qps = self.calculate_derived_qps(conv_pattern)
        tokens_per_second = derived_qps * conv_pattern.avg_sequence_length
        cache_hits_per_second = tokens_per_second * basic_metrics["hit_rate"]
        
        return {
            **basic_metrics,
            "memory_per_token_bytes": memory_per_token,
            "model_memory_gb": model_memory_gb,
            "cache_memory_gb": (basic_metrics["max_cached_tokens"] * memory_per_token) / (1024**3),
            "derived_qps": derived_qps,
            "tokens_per_second": tokens_per_second,
            "cache_hits_per_second": cache_hits_per_second,
            "memory_efficiency": basic_metrics["cache_utilization"]
        }

    def optimize_memory_allocation(self, model_config: ModelConfig,
                                 system_config: SystemConfig,
                                 conv_pattern: ConversationPattern,
                                 target_hit_rate: float = 0.8) -> Dict[str, float]:
        """Optimize memory allocation to achieve target hit rate"""
        
        current_metrics = self.calculate_detailed_metrics(model_config, system_config, conv_pattern)
        
        if current_metrics["hit_rate"] >= target_hit_rate:
            return {"recommended_memory_gb": system_config.available_memory_gb,
                   "current_hit_rate": current_metrics["hit_rate"],
                   "achievable": True}
        
        # Estimate the memory required to achieve the target hit rate
        memory_per_token = self.calculate_kvcache_memory_per_token(model_config)
        avg_tokens_per_conversation = conv_pattern.avg_conversation_length * conv_pattern.avg_sequence_length
        conversation_lifetime = conv_pattern.avg_conversation_length * conv_pattern.within_conversation_interval
        active_conversations = conv_pattern.conversation_arrival_rate * conversation_lifetime
        
        # To achieve the target hit rate, more conversations need to be cached
        required_cached_conversations = active_conversations * target_hit_rate
        required_tokens = required_cached_conversations * avg_tokens_per_conversation
        required_cache_memory = required_tokens * memory_per_token / (1024**3)
        
        required_total_memory = required_cache_memory + model_config.model_size_gb * 1.2
        
        return {
            "recommended_memory_gb": required_total_memory,
            "current_hit_rate": current_metrics["hit_rate"],
            "target_hit_rate": target_hit_rate,
            "additional_memory_needed_gb": max(0, required_total_memory - system_config.available_memory_gb),
            "achievable": required_total_memory <= system_config.available_memory_gb * 2  # Assume at most 2x current memory
        } 
#!/usr/bin/env python3
"""
KVCache命中率计算器使用示例

这个示例展示了如何使用KVCacheCalculator来分析不同配置下的KVCache性能。
"""

from kvcache_calculator import (
    KVCacheCalculator, ModelConfig, SystemConfig, ConversationPattern,
    ModelDtype, KVCacheDtype
)
import json

def main():
    # 初始化计算器
    calculator = KVCacheCalculator()
    
    print("🚀 KVCache命中率计算器示例")
    print("=" * 50)
    
    # 示例1: Llama2-7B 配置
    print("\n📊 示例1: Llama2-7B 模型配置")
    print("-" * 30)
    
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
    
    # 计算详细指标
    metrics = calculator.calculate_detailed_metrics(model_config, system_config, conv_pattern)
    print_metrics(metrics, "Llama2-7B")
    
    # 示例2: FP8优化配置
    print("\n📊 示例2: FP8优化配置 (节省内存)")
    print("-" * 30)
    
    fp8_model_config = ModelConfig(
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=32,
        head_dim=128,
        model_dtype=ModelDtype.FP16,
        kvcache_dtype=KVCacheDtype.FP8,  # 使用FP8来节省内存
        model_size_gb=14.0
    )
    
    fp8_metrics = calculator.calculate_detailed_metrics(fp8_model_config, system_config, conv_pattern)
    print_metrics(fp8_metrics, "FP8优化")
    
    # 比较内存效率
    fp16_memory = calculator.calculate_kvcache_memory_per_token(model_config)
    fp8_memory = calculator.calculate_kvcache_memory_per_token(fp8_model_config)
    memory_saved = (fp16_memory - fp8_memory) / fp16_memory * 100
    
    print(f"\n💾 内存效率对比:")
    print(f"  FP16 KVCache: {fp16_memory:,} 字节/Token")
    print(f"  FP8 KVCache:  {fp8_memory:,} 字节/Token")
    print(f"  内存节省:     {memory_saved:.1f}%")

def print_metrics(metrics, config_name):
    """打印格式化的指标结果"""
    print(f"配置: {config_name}")
    print(f"  📈 KVCache命中率: {metrics['hit_rate']:.1%}")
    print(f"  💾 缓存利用率: {metrics['cache_utilization']:.1%}")
    print(f"  ⚡ 预估延迟减少: {metrics['estimated_latency_reduction']:.1%}")
    print(f"  🗂️ 缓存内存占用: {metrics['cache_memory_gb']:.2f} GB")
    print(f"  🔢 每Token内存: {metrics['memory_per_token_bytes']:.0f} 字节")
    print(f"  💬 可缓存会话数: {metrics['avg_cached_conversations']:.1f}")
    print(f"  🎯 缓存命中次数/秒: {metrics['cache_hits_per_second']:.1f}")

if __name__ == "__main__":
    main() 
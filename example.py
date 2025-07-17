#!/usr/bin/env python3
"""
KVCache Hit Rate Calculator

This example demonstrates how to use the KVCacheCalculator to analyze the 
performance of KVCache under different configurations.
"""

from kvcache_calculator import (
    KVCacheCalculator, ModelConfig, SystemConfig, ConversationPattern,
    ModelDtype, KVCacheDtype
)
import json

def main():
    # Initialize the calculator
    calculator = KVCacheCalculator()
    
    print("🚀 KVCache Hit Rate Calculator Example")
    print("=" * 50)
    
    # Example 1: Mistral-24B Configuration
    print("\n📊 Example 1: Mistral-24B Model Configuration")
    print("-" * 30)
    
    model_config = ModelConfig(
        num_layers=40,
        num_attention_heads=32,
        num_kv_heads=8,
        head_dim=128,
        model_dtype=ModelDtype.FP16,
        kvcache_dtype=KVCacheDtype.FP16,
        model_size_gb=48.0  # 24B * 2 bytes (FP16)
    )
    
    system_config = SystemConfig(
        available_memory_gb=80.0
    )
    
    conv_pattern = ConversationPattern(
        avg_conversation_length=5.0,
        conversation_arrival_rate=2.0,
        within_conversation_interval=30.0,
        avg_sequence_length=1000
    )
    
    # Calculate detailed metrics
    metrics = calculator.calculate_detailed_metrics(model_config, system_config, conv_pattern)
    print_metrics(metrics, "Mistral-24B")
    
    # Example 2: Llama3-8B Configuration
    print("\n📊 Example 2: Llama3-8B Model Configuration")
    print("-" * 30)
    
    llama3_model_config = ModelConfig(
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=32,
        head_dim=128,
        model_dtype=ModelDtype.FP16,
        kvcache_dtype=KVCacheDtype.FP16,
        model_size_gb=16.0  # 8B * 2 bytes (FP16)
    )
    
    llama3_metrics = calculator.calculate_detailed_metrics(llama3_model_config, system_config, conv_pattern)
    print_metrics(llama3_metrics, "Llama3-8B")
    
    # Example 3: Qwen3-32B Configuration
    print("\n📊 Example 3: Qwen3-32B Model Configuration")
    print("-" * 30)
    
    qwen_model_config = ModelConfig(
        num_layers=64,
        num_attention_heads=64,
        num_kv_heads=8,
        head_dim=128,
        model_dtype=ModelDtype.FP16,
        kvcache_dtype=KVCacheDtype.FP16,
        model_size_gb=64.0  # 32B * 2 bytes (FP16)
    )
    
    qwen_metrics = calculator.calculate_detailed_metrics(qwen_model_config, system_config, conv_pattern)
    print_metrics(qwen_metrics, "Qwen3-32B")
    
    # Memory optimization suggestions
    print("\n💡 Memory Optimization Suggestions")
    print("-" * 30)
    
    optimization = calculator.optimize_memory_allocation(
        model_config, system_config, conv_pattern, target_hit_rate=0.85
    )
    
    print(f"Current Configuration (Mistral-24B):")
    print(f"  🎯 Target Hit Rate: 85%")
    print(f"  📊 Current Hit Rate: {optimization['current_hit_rate']:.1%}")
    print(f"  💾 Recommended Memory: {optimization['recommended_memory_gb']:.1f} GB")
    if optimization.get('additional_memory_needed_gb', 0) > 0:
        print(f"  ⚠️ Additional Memory Needed: {optimization['additional_memory_needed_gb']:.1f} GB")
    print(f"  ✅ Target Achievable: {'Yes' if optimization['achievable'] else 'No'}")
    
    # Compare memory efficiency of different models
    print(f"\n💾 Memory Efficiency Comparison:")
    mistral_memory = calculator.calculate_kvcache_memory_per_token(model_config)
    llama3_memory = calculator.calculate_kvcache_memory_per_token(llama3_model_config)
    qwen_memory = calculator.calculate_kvcache_memory_per_token(qwen_model_config)
    
    print(f"  Mistral-24B: {mistral_memory:,} bytes/token")
    print(f"  Llama3-8B:   {llama3_memory:,} bytes/token")
    print(f"  Qwen3-32B:   {qwen_memory:,} bytes/token")

def print_metrics(metrics, config_name):
    """Print formatted metrics results"""
    print(f"Configuration: {config_name}")
    print(f"  📈 KVCache Hit Rate: {metrics['hit_rate']:.1%}")
    print(f"  💾 Cache Utilization: {metrics['cache_utilization']:.1%}")
    print(f"  🔄 System QPS: {metrics['derived_qps']:.1f} req/s")
    print(f"  🗂️ Cache Memory Usage: {metrics['cache_memory_gb']:.2f} GB")
    print(f"  🔢 Memory per Token: {metrics['memory_per_token_bytes']:.0f} bytes")
    print(f"  💬 Average Cached Conversations: {metrics['avg_cached_conversations']:.1f}")
    print(f"  🎯 Cache Hits per Second: {metrics['cache_hits_per_second']:.1f}")
    print(f"  🚀 Tokens per Second: {metrics['tokens_per_second']:.1f}")

if __name__ == "__main__":
    main() 
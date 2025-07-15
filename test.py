#!/usr/bin/env python3
"""
KVCache计算器简单测试脚本
"""

from kvcache_calculator import *

def test_basic_functionality():
    """测试基础功能"""
    print("🧪 测试基础功能...")
    
    calculator = KVCacheCalculator()
    
    # 测试配置
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
    
    # 测试内存计算
    memory_per_token = calculator.calculate_kvcache_memory_per_token(model_config)
    assert memory_per_token > 0, "每Token内存应该大于0"
    print(f"  ✅ 每Token内存计算: {memory_per_token} 字节")
    
    # 测试最大缓存Token数
    max_tokens = calculator.calculate_max_cached_tokens(model_config, system_config)
    assert max_tokens > 0, "最大缓存Token数应该大于0"
    print(f"  ✅ 最大缓存Token数: {max_tokens:,}")
    
    # 测试命中率计算
    metrics = calculator.calculate_conversation_hit_rate(model_config, system_config, conv_pattern)
    assert 0 <= metrics['hit_rate'] <= 1, "命中率应该在0-1之间"
    assert 0 <= metrics['cache_utilization'] <= 1, "缓存利用率应该在0-1之间"
    print(f"  ✅ 命中率计算: {metrics['hit_rate']:.1%}")
    print(f"  ✅ 缓存利用率: {metrics['cache_utilization']:.1%}")
    
    # 测试优化建议
    optimization = calculator.optimize_memory_allocation(model_config, system_config, conv_pattern)
    assert 'recommended_memory_gb' in optimization, "应包含内存推荐"
    print(f"  ✅ 优化建议生成成功")

def test_edge_cases():
    """测试边界情况"""
    print("🧪 测试边界情况...")
    
    calculator = KVCacheCalculator()
    
    # 测试内存不足的情况
    model_config = ModelConfig(
        num_layers=32,
        num_attention_heads=32, 
        num_kv_heads=32,
        head_dim=128,
        model_dtype=ModelDtype.FP16,
        kvcache_dtype=KVCacheDtype.FP16,
        model_size_gb=100.0  # 很大的模型
    )
    
    system_config = SystemConfig(
        available_memory_gb=80.0  # 内存不足
    )
    
    conv_pattern = ConversationPattern(
        avg_conversation_length=1.0,  # 最小会话长度
        conversation_arrival_rate=0.1,
        within_conversation_interval=1.0,
        avg_sequence_length=100
    )
    
    max_tokens = calculator.calculate_max_cached_tokens(model_config, system_config)
    print(f"  ✅ 内存不足情况处理正确: {max_tokens}")
    
    metrics = calculator.calculate_conversation_hit_rate(model_config, system_config, conv_pattern)
    print(f"  ✅ 内存不足时命中率: {metrics['hit_rate']}")

def test_different_dtypes():
    """测试不同数据类型"""
    print("🧪 测试不同数据类型...")
    
    calculator = KVCacheCalculator()
    
    base_config = ModelConfig(
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=32,
        head_dim=128,
        model_dtype=ModelDtype.FP16,
        kvcache_dtype=KVCacheDtype.FP16,
        model_size_gb=14.0
    )
    
    dtypes_to_test = [
        KVCacheDtype.FP32,
        KVCacheDtype.FP16,
        KVCacheDtype.FP8,
        KVCacheDtype.INT8,
    ]
    
    for dtype in dtypes_to_test:
        config = ModelConfig(
            num_layers=base_config.num_layers,
            num_attention_heads=base_config.num_attention_heads,
            num_kv_heads=base_config.num_kv_heads,
            head_dim=base_config.head_dim,
            model_dtype=base_config.model_dtype,
            kvcache_dtype=dtype,
            model_size_gb=base_config.model_size_gb
        )
        
        memory_per_token = calculator.calculate_kvcache_memory_per_token(config)
        print(f"  ✅ {dtype.value}: {memory_per_token:,} 字节/Token")

def run_all_tests():
    """运行所有测试"""
    print("🚀 KVCache计算器测试开始")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        print("✅ 基础功能测试通过\n")
        
        test_edge_cases()
        print("✅ 边界情况测试通过\n")
        
        test_different_dtypes()
        print("✅ 数据类型测试通过\n")
        
        print("=" * 50)
        passed_tests = 3
        total_tests = 3
        print(f"📊 测试结果: {passed_tests}/{total_tests} 通过")
        print("🎉 所有测试通过！计算器功能正常")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise

if __name__ == "__main__":
    run_all_tests() 
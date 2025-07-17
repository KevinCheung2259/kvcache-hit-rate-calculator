#!/usr/bin/env python3
"""
KVCache Calculator Full Test Suite
"""

from kvcache_calculator import *

def test_basic_functionality():
    """Test basic functionality"""
    print("🧪 Test basic functionality...")
    
    calculator = KVCacheCalculator()
    
    # Test configuration - using Mistral-24B configuration
    model_config = ModelConfig(
        num_layers=40,
        num_attention_heads=32,
        num_kv_heads=8,
        head_dim=128,
        model_dtype=ModelDtype.FP16,
        kvcache_dtype=KVCacheDtype.FP16,
        model_size_gb=48.0
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
    
    # Test memory calculation
    memory_per_token = calculator.calculate_kvcache_memory_per_token(model_config)
    assert memory_per_token > 0, "Memory per token should be greater than 0"
    print(f"  ✅ Memory per token calculation: {memory_per_token} bytes")
    
    # Test QPS calculation
    derived_qps = calculator.calculate_derived_qps(conv_pattern)
    expected_qps = conv_pattern.conversation_arrival_rate * conv_pattern.avg_conversation_length
    assert abs(derived_qps - expected_qps) < 0.01, "QPS calculation error"
    print(f"  ✅ QPS calculation: {derived_qps:.1f} req/s")
    
    # Test maximum cached tokens
    max_tokens = calculator.calculate_max_cached_tokens(model_config, system_config)
    assert max_tokens > 0, "Max cached tokens should be greater than 0"
    print(f"  ✅ Max cached tokens: {max_tokens:,}")
    
    # Test hit rate calculation
    hit_rate_metrics = calculator.calculate_conversation_hit_rate(model_config, system_config, conv_pattern)
    assert 0 <= hit_rate_metrics['hit_rate'] <= 1, "Hit rate should be between 0 and 1"
    assert 0 <= hit_rate_metrics['cache_utilization'] <= 1, "Cache utilization should be between 0 and 1"
    print(f"  ✅ Hit rate calculation: {hit_rate_metrics['hit_rate']:.1%}")
    print(f"  ✅ Cache utilization: {hit_rate_metrics['cache_utilization']:.1%}")
    
    # Test detailed metrics calculation
    detailed_metrics = calculator.calculate_detailed_metrics(model_config, system_config, conv_pattern)
    required_fields = ['hit_rate', 'cache_utilization', 'derived_qps', 'tokens_per_second', 
                      'cache_hits_per_second', 'memory_per_token_bytes', 'cache_memory_gb']
    for field in required_fields:
        assert field in detailed_metrics, f"Missing field: {field}"
    print(f"  ✅ Detailed metrics calculation complete")
    
    # Test optimization suggestions
    optimization = calculator.optimize_memory_allocation(model_config, system_config, conv_pattern)
    assert 'recommended_memory_gb' in optimization, "Should contain memory recommendation"
    assert 'achievable' in optimization, "Should contain achievability judgment"
    print(f"  ✅ Optimization suggestions generated successfully")

def test_edge_cases():
    """Test edge cases"""
    print("🧪 Test edge cases...")
    
    calculator = KVCacheCalculator()
    
    # Test memory shortage - using Qwen3-32B configuration but memory shortage
    large_model_config = ModelConfig(
        num_layers=64,
        num_attention_heads=64,
        num_kv_heads=8,
        head_dim=128,
        model_dtype=ModelDtype.FP16,
        kvcache_dtype=KVCacheDtype.FP16,
        model_size_gb=64.0  # 32B model
    )
    
    limited_system_config = SystemConfig(
        available_memory_gb=80.0  # Memory shortage
    )
    
    conv_pattern = ConversationPattern(
        avg_conversation_length=1.0,  # Minimum conversation length
        conversation_arrival_rate=0.1,
        within_conversation_interval=1.0,
        avg_sequence_length=100
    )
    
    max_tokens = calculator.calculate_max_cached_tokens(large_model_config, limited_system_config)
    print(f"  ✅ Memory shortage handling: {max_tokens}")
    
    metrics = calculator.calculate_conversation_hit_rate(large_model_config, limited_system_config, conv_pattern)
    print(f"  ✅ Hit rate when memory shortage: {metrics['hit_rate']:.1%}")
    
    # Test extreme high load
    high_load_pattern = ConversationPattern(
        avg_conversation_length=10.0,
        conversation_arrival_rate=100.0,  # Extremely high arrival rate
        within_conversation_interval=1.0,  # Extremely short interval
        avg_sequence_length=2000
    )
    
    high_load_metrics = calculator.calculate_conversation_hit_rate(
        large_model_config, limited_system_config, high_load_pattern
    )
    print(f"  ✅ High load handling: {high_load_metrics['hit_rate']:.1%}")

def test_different_dtypes():
    """Test different data types"""
    print("🧪 Test different data types...")
    
    calculator = KVCacheCalculator()
    
    base_config = ModelConfig(
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=32,
        head_dim=128,
        model_dtype=ModelDtype.FP16,
        kvcache_dtype=KVCacheDtype.FP16,
        model_size_gb=16.0
    )
    
    # 测试剩余的数据类型（移除FP8）
    dtypes_to_test = [
        KVCacheDtype.FP32,
        KVCacheDtype.FP16,
        KVCacheDtype.BF16,
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
        print(f"  ✅ {dtype.value}: {memory_per_token:,} bytes/token")

def test_optimization_scenarios():
    """Test optimization scenarios"""
    print("🧪 Test optimization scenarios...")
    
    calculator = KVCacheCalculator()
    
    # Test configuration
    model_config = ModelConfig(
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=32,
        head_dim=128,
        model_dtype=ModelDtype.FP16,
        kvcache_dtype=KVCacheDtype.FP16,
        model_size_gb=16.0
    )
    
    system_config = SystemConfig(available_memory_gb=80.0)
    conv_pattern = ConversationPattern(
        avg_conversation_length=5.0,
        conversation_arrival_rate=2.0,
        within_conversation_interval=30.0,
        avg_sequence_length=1000
    )
    
    # Test different target hit rates
    target_rates = [0.5, 0.7, 0.8, 0.9, 0.95]
    
    for target_rate in target_rates:
        optimization = calculator.optimize_memory_allocation(
            model_config, system_config, conv_pattern, target_hit_rate=target_rate
        )
        print(f"  ✅ Target hit rate {target_rate:.0%}: Recommended memory {optimization['recommended_memory_gb']:.1f} GB")

def run_all_tests():
    """Run all tests"""
    print("🚀 KVCache Calculator Test Start")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        print("✅ Basic functionality test passed\n")
        
        test_edge_cases()
        print("✅ Edge cases test passed\n")
        
        test_different_dtypes()
        print("✅ Data types test passed\n")
        
        test_optimization_scenarios()
        print("✅ Optimization scenarios test passed\n")
        
        print("=" * 50)
        passed_tests = 4
        total_tests = 4
        print(f"📊 Test results: {passed_tests}/{total_tests} passed")
        print("🎉 All tests passed! Calculator functionality is normal")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    run_all_tests() 
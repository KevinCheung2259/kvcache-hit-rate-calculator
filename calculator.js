// KVCache命中率计算器 JavaScript实现

class KVCacheCalculator {
    constructor() {
        this.dtypeBytes = {
            'fp32': 4,
            'fp16': 2,
            'bf16': 2,
            'fp8': 1,
            'int8': 1,
            'int4': 0.5
        };
    }

    calculateModelMemoryGb(modelConfig) {
        // 计算模型参数内存占用
        // 模型内存 = 参数数量 × 数据类型字节数
        const bytesPerParam = this.dtypeBytes[modelConfig.modelDtype];
        const modelMemoryBytes = modelConfig.numParams * bytesPerParam;
        const modelMemoryGb = modelMemoryBytes / (1024**3);
        // 包含运行时开销，通常为1.2-1.5倍
        return modelMemoryGb * 1.3;
    }

    calculateKVCacheMemoryPerToken(modelConfig) {
        // KVCache包含Key和Value，每层都有
        // 内存 = 2 (K+V) * num_layers * num_kv_heads * head_dim * dtype_bytes
        const bytesPerElement = this.dtypeBytes[modelConfig.kvcacheDtype];
        const memoryPerToken = 2 * modelConfig.numLayers * 
                              modelConfig.numKvHeads * 
                              modelConfig.headDim * 
                              bytesPerElement;
        return memoryPerToken;
    }

    calculateDerivedQps(convPattern) {
        // QPS = 每秒的请求数量，与token长度无关
        // 基于Little's Law：
        // 系统QPS = 会话到达率 × 平均会话长度（每个会话的请求数）
        const derivedQps = convPattern.conversationArrivalRate * convPattern.avgConversationLength;
        return derivedQps;
    }

    calculateMaxCachedTokens(modelConfig, systemConfig) {
        // 计算最大可缓存的token数量
        const memoryPerToken = this.calculateKVCacheMemoryPerToken(modelConfig);
        
        // 计算模型内存：参数数量 × 数据类型字节数
        const bytesPerParam = this.dtypeBytes[modelConfig.modelDtype];
        const modelMemoryBytes = modelConfig.numParams * bytesPerParam;
        const modelMemoryGb = (modelMemoryBytes / (1024**3)) * 1.3; // 包含运行时开销
        
        // 可用于KVCache的内存 = 总内存 - 模型内存
        const availableForCache = (systemConfig.availableMemoryGb * 1024**3 - 
                                  modelMemoryGb * 1024**3);
        
        if (availableForCache <= 0) {
            return 0;
        }
        
        const maxTokens = Math.floor(availableForCache / memoryPerToken);
        return maxTokens;
    }

    calculateConversationHitRate(modelConfig, systemConfig, convPattern) {
        // 计算会话级别的命中率
        const maxCachedTokens = this.calculateMaxCachedTokens(modelConfig, systemConfig);
        
        if (maxCachedTokens <= 0) {
            return {
                hitRate: 0.0,
                avgCachedConversations: 0.0,
                cacheUtilization: 0.0,
                maxCachedTokens: 0,
                activeConversations: 0
            };
        }

        // 估算平均每个会话的token数
        const avgTokensPerConversation = convPattern.avgConversationLength * convPattern.avgSequenceLength;
        
        // 可以缓存的会话数量
        const maxCachedConversations = maxCachedTokens / avgTokensPerConversation;
        
        // 会话到达率和生存时间建模
        const conversationLifetime = convPattern.avgConversationLength * convPattern.withinConversationInterval;
        
        // 使用Little's Law: 系统中的平均会话数 = 到达率 × 平均停留时间
        const activeConversations = convPattern.conversationArrivalRate * conversationLifetime;
        
        let hitRate;
        if (activeConversations <= maxCachedConversations) {
            // 所有活跃会话都能被缓存
            hitRate = 1.0 - (1.0 / convPattern.avgConversationLength);
        } else {
            // 部分会话被缓存，使用概率模型
            const cacheRatio = maxCachedConversations / activeConversations;
            const intraConversationHit = 1.0 - (1.0 / convPattern.avgConversationLength);
            const interConversationHit = cacheRatio;
            hitRate = intraConversationHit * interConversationHit;
        }
        
        const cacheUtilization = maxCachedConversations > 0 ? 
            Math.min(activeConversations / maxCachedConversations, 1.0) : 0;
        
        return {
            hitRate: Math.max(0.0, Math.min(1.0, hitRate)),
            avgCachedConversations: Math.min(activeConversations, maxCachedConversations),
            cacheUtilization: cacheUtilization,
            maxCachedTokens: maxCachedTokens,
            activeConversations: activeConversations
        };
    }

    calculateDetailedMetrics(modelConfig, systemConfig, convPattern) {
        // 计算详细的性能指标
        const basicMetrics = this.calculateConversationHitRate(modelConfig, systemConfig, convPattern);
        
        // 计算内存使用详情
        const memoryPerToken = this.calculateKVCacheMemoryPerToken(modelConfig);
        
        // 计算模型内存：参数数量 × 数据类型字节数
        const bytesPerParam = this.dtypeBytes[modelConfig.modelDtype];
        const modelMemoryBytes = modelConfig.numParams * bytesPerParam;
        const modelMemoryGb = (modelMemoryBytes / (1024**3)) * 1.3; // 包含运行时开销
        
        // 计算性能提升 - QPS从会话参数推导
        const derivedQps = this.calculateDerivedQps(convPattern);
        const tokensPerSecond = derivedQps * convPattern.avgSequenceLength;
        const cacheHitsPerSecond = tokensPerSecond * basicMetrics.hitRate;
        
        // 估算延迟改善（缓存命中可以减少计算时间）
        const computeReductionFactor = 0.3;  // 假设缓存命中减少30%计算时间
        const avgLatencyReduction = basicMetrics.hitRate * computeReductionFactor;
        
        return {
            ...basicMetrics,
            memoryPerTokenBytes: memoryPerToken,
            modelMemoryGb: modelMemoryGb,
            cacheMemoryGb: (basicMetrics.maxCachedTokens * memoryPerToken) / (1024**3),
            derivedQps: derivedQps,
            tokensPerSecond: tokensPerSecond,
            cacheHitsPerSecond: cacheHitsPerSecond,
            estimatedLatencyReduction: avgLatencyReduction,
            memoryEfficiency: basicMetrics.cacheUtilization
        };
    }
}

// 全局计算器实例
const calculator = new KVCacheCalculator();

// 全局图表变量
let hitRateChart = null;

// 预设配置
const presets = {
    'llama2-7b': {
        numLayers: 32,
        numKvHeads: 32,
        headDim: 128,
        numParams: 7,  // 7B参数，单位亿
        modelDtype: 'fp16',
        kvcacheDtype: 'fp16'
    },
    'llama2-13b': {
        numLayers: 40,
        numKvHeads: 40,
        headDim: 128,
        numParams: 13, // 13B参数，单位亿
        modelDtype: 'fp16',
        kvcacheDtype: 'fp16'
    },
    'chatglm-6b': {
        numLayers: 28,
        numKvHeads: 2,
        headDim: 128,
        numParams: 6,  // 6B参数，单位亿
        modelDtype: 'fp16',
        kvcacheDtype: 'fp16'
    },
    'fp8-optimized': {
        numLayers: 32,
        numKvHeads: 32,
        headDim: 128,
        numParams: 7,  // 7B参数，单位亿
        modelDtype: 'fp8',
        kvcacheDtype: 'fp8'
    },
    'custom-large': {
        numLayers: 80,
        numKvHeads: 64,
        headDim: 128,
        numParams: 70, // 70B参数，单位亿
        modelDtype: 'fp16',
        kvcacheDtype: 'fp16'
    }
};

// 工具函数
function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function formatNumber(num) {
    return num.toLocaleString();
}

// 加载预设配置
function loadPreset(presetName) {
    const preset = presets[presetName];
    if (!preset) return;

    document.getElementById('num-layers').value = preset.numLayers;
    document.getElementById('num-kv-heads').value = preset.numKvHeads;
    document.getElementById('head-dim').value = preset.headDim;
    document.getElementById('num-params').value = preset.numParams;
    document.getElementById('model-dtype').value = preset.modelDtype;
    document.getElementById('kvcache-dtype').value = preset.kvcacheDtype;
    
    // 自动更新图表
    updateChart();
}

// 获取表单数据
function getFormData() {
    return {
        modelConfig: {
            numLayers: parseInt(document.getElementById('num-layers').value),
            numKvHeads: parseInt(document.getElementById('num-kv-heads').value),
            headDim: parseInt(document.getElementById('head-dim').value),
            numParams: parseFloat(document.getElementById('num-params').value) * 1e9, // 转换为实际参数数量
            modelDtype: document.getElementById('model-dtype').value,
            kvcacheDtype: document.getElementById('kvcache-dtype').value
        },
        systemConfig: {
            availableMemoryGb: parseFloat(document.getElementById('available-memory').value)
        },
        convPattern: {
            avgConversationLength: parseFloat(document.getElementById('avg-conv-length').value),
            conversationArrivalRate: parseFloat(document.getElementById('conv-arrival-rate').value),
            withinConversationInterval: parseFloat(document.getElementById('within-conv-interval').value),
            avgSequenceLength: parseInt(document.getElementById('avg-sequence-length').value)
        }
    };
}

// 主计算函数
function calculateHitRate() {
    try {
        const data = getFormData();
        const metrics = calculator.calculateDetailedMetrics(
            data.modelConfig, 
            data.systemConfig, 
            data.convPattern
        );
        
        displayResults(metrics);
        generateOptimizationTips(metrics, data);
        drawHitRateChart(); // 绘制图表
        
    } catch (error) {
        console.error('计算错误:', error);
        alert('计算出错: ' + error.message);
    }
}

// 显示结果
function displayResults(metrics) {
    // 主要指标
    document.getElementById('hit-rate').textContent = (metrics.hitRate * 100).toFixed(1);
    document.getElementById('cache-utilization').textContent = (metrics.cacheUtilization * 100).toFixed(1);
    document.getElementById('latency-reduction').textContent = (metrics.estimatedLatencyReduction * 100).toFixed(1);
    document.getElementById('cache-memory').textContent = metrics.cacheMemoryGb.toFixed(2);
    
    // 详细指标
    document.getElementById('derived-qps').textContent = metrics.derivedQps.toFixed(1);
    document.getElementById('memory-per-token').textContent = formatBytes(metrics.memoryPerTokenBytes);
    document.getElementById('max-cached-tokens').textContent = formatNumber(metrics.maxCachedTokens);
    document.getElementById('active-conversations').textContent = metrics.activeConversations.toFixed(1);
    document.getElementById('cache-hits-per-second').textContent = metrics.cacheHitsPerSecond.toFixed(1);
    document.getElementById('model-memory').textContent = metrics.modelMemoryGb.toFixed(1) + ' GB';
    document.getElementById('cached-conversations').textContent = metrics.avgCachedConversations.toFixed(1);
}

// 绘制命中率-内存关系图表
function drawHitRateChart() {
    const data = getFormData();
    const currentMemory = data.systemConfig.availableMemoryGb;
    
    // 计算模型内存
    const bytesPerParam = calculator.dtypeBytes[data.modelConfig.modelDtype];
    const modelMemoryBytes = data.modelConfig.numParams * bytesPerParam;
    const modelMemoryGb = (modelMemoryBytes / (1024**3)) * 1.3; // 包含运行时开销
    
    // 生成内存范围（从模型内存的1.5倍到当前内存的3倍）
    const memoryRange = [];
    const hitRates = [];
    
    const minMemory = Math.max(modelMemoryGb * 1.5, currentMemory * 0.3);
    const maxMemory = currentMemory * 3;
    const step = (maxMemory - minMemory) / 20; // 20个数据点
    
    for (let memory = minMemory; memory <= maxMemory; memory += step) {
        const testSystemConfig = { ...data.systemConfig, availableMemoryGb: memory };
        const metrics = calculator.calculateDetailedMetrics(
            data.modelConfig,
            testSystemConfig,
            data.convPattern
        );
        
        memoryRange.push(memory);
        hitRates.push(metrics.hitRate * 100);
    }
    
    // 获取图表canvas
    const ctx = document.getElementById('hit-rate-chart').getContext('2d');
    
    // 销毁之前的图表
    if (hitRateChart) {
        hitRateChart.destroy();
    }
    
    // 创建新图表
    hitRateChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: memoryRange.map(m => m.toFixed(0) + 'GB'),
            datasets: [
                {
                    label: '命中率 (%)',
                    data: hitRates,
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 2,
                    pointHoverRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                title: {
                    display: true,
                    text: 'KVCache命中率随内存变化',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                },
                legend: {
                    position: 'top',
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: '可用内存 (GB)',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: '命中率 (%)',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    },
                    min: 0,
                    max: 100
                }
            }
        }
    });
}

// 生成优化建议
function generateOptimizationTips(metrics, data) {
    const container = document.getElementById('optimization-results');
    let html = '';
    
    if (metrics.hitRate < 0.5) {
        html += `
            <div class="optimization-tip warning">
                ⚠️ 建议增加内存或优化会话模式以提高命中率
            </div>
        `;
    }
    
    if (metrics.cacheUtilization < 0.3) {
        html += `
            <div class="optimization-tip info">
                💡 缓存利用率较低，考虑减少内存分配或增加负载
            </div>
        `;
    }
    
    if (data.modelConfig.kvcacheDtype === 'fp32' || data.modelConfig.kvcacheDtype === 'fp16') {
        html += `
            <div class="optimization-tip success">
                🎯 考虑使用FP8量化来减少50-75%的内存使用量
            </div>
        `;
    }
    
    if (metrics.activeConversations > metrics.avgCachedConversations * 2) {
        html += `
            <div class="optimization-tip warning">
                📈 检测到高会话负载，考虑水平扩展
            </div>
        `;
    }
    
    if (!html) {
        html = '<p class="optimization-tip success">✅ 当前配置较为合理</p>';
    }
    
    container.innerHTML = html;
}

// 初始化页面
document.addEventListener('DOMContentLoaded', function() {
    // 设置事件监听器
    document.getElementById('calculate-btn').addEventListener('click', calculateHitRate);
    
    // 为所有输入框添加自动更新监听器
    const inputIds = [
        'num-layers', 'num-kv-heads', 'head-dim', 'num-params', 'model-dtype', 'kvcache-dtype', 
        'available-memory', 'avg-conv-length', 'conv-arrival-rate', 
        'within-conv-interval', 'avg-sequence-length'
    ];
    
    inputIds.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.addEventListener('input', function() {
                // 添加小延迟避免频繁更新
                clearTimeout(element.updateTimeout);
                element.updateTimeout = setTimeout(() => {
                    updateChart();
                }, 300);
            });
        }
    });
    
    // 默认加载Llama2-7B配置
    loadPreset('llama2-7b');
});

// 自动更新所有内容（结果、指标、图表）
function updateChart() {
    try {
        const data = getFormData();
        const metrics = calculator.calculateDetailedMetrics(
            data.modelConfig, 
            data.systemConfig, 
            data.convPattern
        );
        
        displayResults(metrics);
        generateOptimizationTips(metrics, data);
        drawHitRateChart();
    } catch (error) {
        console.error('更新内容错误:', error);
    }
} 
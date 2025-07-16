// KVCache Hit Rate Calculator JavaScript implementation

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
        // Calculate the memory occupied by model parameters
        // Model memory = number of parameters × data type bytes
        const bytesPerParam = this.dtypeBytes[modelConfig.modelDtype];
        const modelMemoryBytes = modelConfig.numParams * bytesPerParam;
        const modelMemoryGb = modelMemoryBytes / (1024**3);
        // Includes runtime overhead, usually 1.2-1.5 times
        return modelMemoryGb * 1.3;
    }

    calculateKVCacheMemoryPerToken(modelConfig) {
        // KVCache contains Key and Value, each layer has
        // Memory = 2 (K+V) * num_layers * num_kv_heads * head_dim * dtype_bytes
        const bytesPerElement = this.dtypeBytes[modelConfig.kvcacheDtype];
        const memoryPerToken = 2 * modelConfig.numLayers * 
                              modelConfig.numKvHeads * 
                              modelConfig.headDim * 
                              bytesPerElement;
        return memoryPerToken;
    }

    calculateDerivedQps(convPattern) {
        // QPS = number of requests per second, independent of token length
        // Based on Little's Law:
        // System QPS = conversation arrival rate × average conversation length (number of requests per conversation)
        const derivedQps = convPattern.conversationArrivalRate * convPattern.avgConversationLength;
        return derivedQps;
    }

    calculateMaxCachedTokens(modelConfig, systemConfig) {
        // Calculate the maximum number of tokens that can be cached
        const memoryPerToken = this.calculateKVCacheMemoryPerToken(modelConfig);
        
        // Calculate model memory: number of parameters × data type bytes
        const bytesPerParam = this.dtypeBytes[modelConfig.modelDtype];
        const modelMemoryBytes = modelConfig.numParams * bytesPerParam;
        const modelMemoryGb = (modelMemoryBytes / (1024**3)) * 1.3; // Includes runtime overhead
        
        // Available memory for KVCache = total memory - model memory
        const availableForCache = (systemConfig.availableMemoryGb * 1024**3 - 
                                  modelMemoryGb * 1024**3);
        
        if (availableForCache <= 0) {
            return 0;
        }
        
        const maxTokens = Math.floor(availableForCache / memoryPerToken);
        return maxTokens;
    }

    calculateConversationHitRate(modelConfig, systemConfig, convPattern) {
        // Calculate the hit rate at the conversation level
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

        // Estimate the average number of tokens per conversation
        const avgTokensPerConversation = convPattern.avgConversationLength * convPattern.avgSequenceLength;
        
        // Number of conversations that can be cached
        const maxCachedConversations = maxCachedTokens / avgTokensPerConversation;
        
        // Model conversation arrival rate and survival time
        const conversationLifetime = convPattern.avgConversationLength * convPattern.withinConversationInterval;
        
        // Using Little's Law: average number of conversations in the system = arrival rate × average stay time
        const activeConversations = convPattern.conversationArrivalRate * conversationLifetime;
        
        let hitRate;
        if (activeConversations <= maxCachedConversations) {
            // All active conversations can be cached
            hitRate = 1.0 - (1.0 / convPattern.avgConversationLength);
        } else {
            // Some conversations are cached, use probabilistic model
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
        // Calculate detailed performance metrics
        const basicMetrics = this.calculateConversationHitRate(modelConfig, systemConfig, convPattern);
        
        // Calculate memory usage details
        const memoryPerToken = this.calculateKVCacheMemoryPerToken(modelConfig);
        
        // Calculate model memory: number of parameters × data type bytes
        const bytesPerParam = this.dtypeBytes[modelConfig.modelDtype];
        const modelMemoryBytes = modelConfig.numParams * bytesPerParam;
        const modelMemoryGb = (modelMemoryBytes / (1024**3)) * 1.3; // Includes runtime overhead
        
        // Calculate performance improvement - QPS derived from conversation parameters
        const derivedQps = this.calculateDerivedQps(convPattern);
        const tokensPerSecond = derivedQps * convPattern.avgSequenceLength;
        const cacheHitsPerSecond = tokensPerSecond * basicMetrics.hitRate;
        
        return {
            ...basicMetrics,
            memoryPerTokenBytes: memoryPerToken,
            modelMemoryGb: modelMemoryGb,
            cacheMemoryGb: (basicMetrics.maxCachedTokens * memoryPerToken) / (1024**3),
            derivedQps: derivedQps,
            tokensPerSecond: tokensPerSecond,
            cacheHitsPerSecond: cacheHitsPerSecond,
            derivedQpsForDisplay: derivedQps,
            memoryEfficiency: basicMetrics.cacheUtilization
        };
    }
}

// Global calculator instance
const calculator = new KVCacheCalculator();

// Global chart variable
let hitRateChart = null;

// Preset configurations
const presets = {
    'mistral-24B': {
        numLayers: 40,
        numKvHeads: 8,
        headDim: 128,
        numParams: 24,
        modelDtype: 'fp16',
        kvcacheDtype: 'fp16'
    },
    'llama3-8B': {
        numLayers: 32,
        numKvHeads: 32,
        headDim: 128,
        numParams: 8, 
        modelDtype: 'fp16',
        kvcacheDtype: 'fp16'
    },
    'qwen3-32B': {
        numLayers: 64,
        numKvHeads: 8,
        headDim: 128,
        numParams: 32,
        modelDtype: 'fp16',
        kvcacheDtype: 'fp16'
    },
    
};

// Utility functions
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

// Load preset configurations
function loadPreset(presetName) {
    const preset = presets[presetName];
    if (!preset) return;

    document.getElementById('num-layers').value = preset.numLayers;
    document.getElementById('num-kv-heads').value = preset.numKvHeads;
    document.getElementById('head-dim').value = preset.headDim;
    document.getElementById('num-params').value = preset.numParams;
    document.getElementById('model-dtype').value = preset.modelDtype;
    document.getElementById('kvcache-dtype').value = preset.kvcacheDtype;
    
    // Automatically update the chart
    updateChart();
}

// Get form data
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

// Main calculation function
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
        console.error('Calculation error:', error);
        alert('Calculation error: ' + error.message);
    }
}

// Display results
function displayResults(metrics) {
    // Main metrics
    document.getElementById('hit-rate').textContent = (metrics.hitRate * 100).toFixed(1);
    document.getElementById('cache-utilization').textContent = (metrics.cacheUtilization * 100).toFixed(1);
    document.getElementById('derived-qps').textContent = metrics.derivedQps.toFixed(1);
    document.getElementById('cache-memory').textContent = metrics.cacheMemoryGb.toFixed(2);
    document.getElementById('memory-per-token').textContent = formatBytes(metrics.memoryPerTokenBytes);
    document.getElementById('max-cached-tokens').textContent = formatNumber(metrics.maxCachedTokens);
    document.getElementById('active-conversations').textContent = metrics.activeConversations.toFixed(1);
    document.getElementById('cache-hits-per-second').textContent = metrics.cacheHitsPerSecond.toFixed(1);
    document.getElementById('model-memory').textContent = metrics.modelMemoryGb.toFixed(1) + ' GB';
    document.getElementById('cached-conversations').textContent = metrics.avgCachedConversations.toFixed(1);
}

// Draw hit rate vs memory chart
function drawHitRateChart() {
    const data = getFormData();
    const currentMemory = data.systemConfig.availableMemoryGb;
    
    // Calculate model memory
    const bytesPerParam = calculator.dtypeBytes[data.modelConfig.modelDtype];
    const modelMemoryBytes = data.modelConfig.numParams * bytesPerParam;
    const modelMemoryGb = (modelMemoryBytes / (1024**3)) * 1.3; // Includes runtime overhead
    
    // Generate memory range (from 1.5x model memory to 3x current memory)
    const memoryRange = [];
    const hitRates = [];
    
    const minMemory = Math.max(modelMemoryGb * 1.5, currentMemory * 0.3);
    const maxMemory = currentMemory * 3;
    const step = (maxMemory - minMemory) / 20; // 20 data points
    
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
    
    // Get chart canvas
    const ctx = document.getElementById('hit-rate-chart').getContext('2d');
    
    // Destroy previous chart
    if (hitRateChart) {
        hitRateChart.destroy();
    }
    
    // Create new chart
    hitRateChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: memoryRange.map(m => m.toFixed(0) + 'GB'),
            datasets: [
                {
                    label: 'Hit Rate (%)',
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
                    text: 'KVCache Hit Rate vs Memory',
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
                        text: 'Available Memory (GB)',
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
                        text: 'Hit Rate (%)',
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

// Generate optimization tips
function generateOptimizationTips(metrics, data) {
    const container = document.getElementById('optimization-results');
    let html = '';
    
    if (metrics.hitRate < 0.5) {
        html += `
            <div class="optimization-tip warning">
                ⚠️ Consider increasing memory or decreasing conversation arrival rate to improve hit rate
            </div>
        `;
    }
    
    if (metrics.cacheUtilization < 0.3) {
        html += `
            <div class="optimization-tip info">
                💡 Low cache utilization, consider reducing memory allocation or increasing load
            </div>
        `;
    }
    
    if (data.modelConfig.kvcacheDtype === 'fp32' || data.modelConfig.kvcacheDtype === 'fp16') {
        html += `
            <div class="optimization-tip success">
                🎯 Consider using FP8 quantization to reduce memory usage
            </div>
        `;
    }
    
    if (metrics.activeConversations > metrics.avgCachedConversations * 2) {
        html += `
            <div class="optimization-tip warning">
                📈 High conversation load detected, consider horizontal scaling
            </div>
        `;
    }
    
    if (!html) {
        html = '<p class="optimization-tip success">✅ Current configuration is reasonable</p>';
    }
    
    container.innerHTML = html;
}

// Initialize the page
document.addEventListener('DOMContentLoaded', function() {
    // Set event listeners
    document.getElementById('calculate-btn').addEventListener('click', calculateHitRate);
    
    // Add auto-update listeners to all input fields
    const inputIds = [
        'num-layers', 'num-kv-heads', 'head-dim', 'num-params', 'model-dtype', 'kvcache-dtype', 
        'available-memory', 'avg-conv-length', 'conv-arrival-rate', 
        'within-conv-interval', 'avg-sequence-length'
    ];
    
    inputIds.forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            element.addEventListener('input', function() {
                // Add a small delay to avoid frequent updates
                clearTimeout(element.updateTimeout);
                element.updateTimeout = setTimeout(() => {
                    updateChart();
                }, 300);
            });
        }
    });
    
    // Default load mistral-24B configuration
    loadPreset('mistral-24B');
});

// Automatically update all content (results, metrics, charts)
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
        console.error('Update content error:', error);
    }
} 
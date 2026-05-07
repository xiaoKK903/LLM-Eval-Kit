#!/usr/bin/env python3
"""
测试多模型适配器功能
"""

import asyncio
from llm_eval_kit.adapters.multi_provider_adapter import MultiProviderAdapter

async def test_multi_provider():
    """测试多模型适配器功能"""
    print("🧪 测试多模型适配器功能")
    print("=" * 60)
    
    # 测试支持的模型列表
    supported_models = MultiProviderAdapter.get_supported_models()
    print("📊 支持的模型列表:")
    for provider, models in supported_models.items():
        print(f"  {provider.upper()}: {', '.join(models)}")
    
    print("\n💰 模型定价信息:")
    from llm_eval_kit.reporter.comparator import ModelComparator
    comparator = ModelComparator()
    
    # 显示部分模型的定价
    test_models = ["qwen-turbo", "gpt-3.5-turbo", "glm-4", "ernie-bot", "spark-v3"]
    for model in test_models:
        pricing = comparator.pricing.get(model, comparator.pricing["default"])
        print(f"  {model}: 输入 ¥{pricing['input']}/百万token, 输出 ¥{pricing['output']}/百万token")
    
    print("\n🔧 适配器初始化测试:")
    
    # 测试不同模型的适配器初始化
    test_configs = [
        {
            "model": "qwen-turbo",
            "api_key": "test-key",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
        },
        {
            "model": "gpt-3.5-turbo", 
            "api_key": "test-key",
            "base_url": "https://api.openai.com/v1"
        },
        {
            "model": "glm-4",
            "api_key": "test-key", 
            "base_url": "https://open.bigmodel.cn/api/paas/v4"
        }
    ]
    
    for config in test_configs:
        try:
            adapter = MultiProviderAdapter(config)
            print(f"  ✅ {config['model']} 适配器初始化成功 (provider: {adapter.provider})")
            await adapter.close()
        except Exception as e:
            print(f"  ❌ {config['model']} 适配器初始化失败: {e}")
    
    print("\n📈 成本计算测试:")
    
    # 测试成本计算
    test_cases = [
        ("qwen-turbo", 1000, 500),
        ("gpt-4", 1000, 500),
        ("glm-4", 1000, 500),
        ("unknown-model", 1000, 500)
    ]
    
    for model, prompt_tokens, completion_tokens in test_cases:
        cost = comparator._calculate_cost(model, prompt_tokens, completion_tokens)
        print(f"  {model}: {prompt_tokens}输入 + {completion_tokens}输出 = ¥{cost:.6f}")
    
    print("\n🎯 模型提供商配置测试:")
    
    # 测试提供商配置
    providers = ["openai", "qwen", "deepseek", "zhipu", "ernie", "spark"]
    for provider in providers:
        config = MultiProviderAdapter.get_provider_config(provider)
        if config:
            print(f"  ✅ {provider}: 基础URL = {config['base_url']}")
        else:
            print(f"  ❌ {provider}: 配置不存在")
    
    print("\n" + "=" * 60)
    print("✅ 多模型适配器测试完成！")

if __name__ == "__main__":
    asyncio.run(test_multi_provider())
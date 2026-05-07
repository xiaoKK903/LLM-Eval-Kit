"""
Complete list of available Chinese LLM models for testing.

This file provides a comprehensive list of all supported models
from major Chinese LLM providers.
"""

# Complete list of Chinese LLM models
COMPLETE_MODELS_LIST = {
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "QWEN_API_KEY",
        "models": [
            {"name": "qwen-turbo", "description": "千问Turbo(高速版)", "recommended": True},
            {"name": "qwen-plus", "description": "千问Plus(增强版)", "recommended": True},
            {"name": "qwen-max", "description": "千问Max(最强版)", "recommended": True},
            {"name": "qwen-7b-chat", "description": "千问7B对话模型"},
            {"name": "qwen-14b-chat", "description": "千问14B对话模型"},
            {"name": "qwen-72b-chat", "description": "千问72B对话模型"},
        ]
    },
    
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY", 
        "models": [
            {"name": "deepseek-chat", "description": "DeepSeek对话模型", "recommended": True},
            {"name": "deepseek-coder", "description": "DeepSeek代码模型", "recommended": True},
            {"name": "deepseek-v4-pro", "description": "DeepSeek V4专业版", "recommended": True},
            {"name": "deepseek-v4-flash", "description": "DeepSeek V4快速版"},
            {"name": "deepseek-reasoner", "description": "DeepSeek推理模型"},
        ]
    },
    
    "baichuan": {
        "base_url": "https://api.baichuan-ai.com/v1",
        "api_key_env": "BAICHUAN_API_KEY",
        "models": [
            {"name": "Baichuan-Turbo", "description": "百川Turbo模型"},
            {"name": "Baichuan-Pro", "description": "百川专业模型"},
        ]
    },
    
    "chatglm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4", 
        "api_key_env": "CHATGLM_API_KEY",
        "models": [
            {"name": "chatglm-turbo", "description": "ChatGLM Turbo模型"},
            {"name": "chatglm-pro", "description": "ChatGLM专业模型"},
        ]
    }
}

def get_recommended_models():
    """Get only the recommended models for quick testing."""
    recommended = []
    
    for provider, config in COMPLETE_MODELS_LIST.items():
        for model in config["models"]:
            if model.get("recommended", False):
                recommended.append({
                    "name": model["name"],
                    "description": model["description"],
                    "base_url": config["base_url"],
                    "api_key_env": config["api_key_env"],
                    "provider": provider
                })
    
    return recommended

def get_all_models():
    """Get all available models."""
    all_models = []
    
    for provider, config in COMPLETE_MODELS_LIST.items():
        for model in config["models"]:
            all_models.append({
                "name": model["name"],
                "description": model["description"],
                "base_url": config["base_url"],
                "api_key_env": config["api_key_env"],
                "provider": provider,
                "recommended": model.get("recommended", False)
            })
    
    return all_models

def print_model_summary():
    """Print a summary of available models."""
    print("🚀 中国LLM模型完整列表")
    print("=" * 60)
    
    total_models = 0
    for provider, config in COMPLETE_MODELS_LIST.items():
        provider_name = {
            "qwen": "阿里云通义千问",
            "deepseek": "深度求索DeepSeek", 
            "baichuan": "百川智能",
            "chatglm": "智谱AI ChatGLM"
        }.get(provider, provider)
        
        print(f"\n📊 {provider_name}:")
        for model in config["models"]:
            total_models += 1
            rec_mark = "⭐ " if model.get("recommended", False) else "  "
            print(f"   {rec_mark}{model['name']:20} - {model['description']}")
    
    print(f"\n📈 总计: {total_models} 个模型")
    
    recommended = get_recommended_models()
    print(f"⭐ 推荐测试: {len(recommended)} 个核心模型")
    
    print("\n💡 使用建议:")
    print("   1. 先测试推荐模型 (标记⭐)")
    print("   2. 根据业务需求选择特定模型")
    print("   3. 使用 examples/basic_qwen_deepseek.py 进行评测")

if __name__ == "__main__":
    print_model_summary()
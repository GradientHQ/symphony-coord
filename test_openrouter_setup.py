#!/usr/bin/env python3
"""
测试 OpenRouter 配置是否正确
"""
import os
import sys
import yaml

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_api_key():
    """测试 API key 是否设置"""
    print("=" * 60)
    print("1. 检查 API Key")
    print("=" * 60)
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if api_key:
        print(f"✅ OPENROUTER_API_KEY is set (length: {len(api_key)})")
        print(f"   前缀: {api_key[:10]}...")
        return True
    else:
        print("❌ OPENROUTER_API_KEY not set")
        print("   请运行: export OPENROUTER_API_KEY='your_key'")
        return False

def test_code_support():
    """测试代码支持"""
    print("\n" + "=" * 60)
    print("2. 检查代码支持")
    print("=" * 60)
    
    try:
        from agents.agent import _is_openrouter_spec, _parse_openrouter_spec
        
        test_cases = [
            "openrouter:openai/gpt-4o-mini",
            "openrouter:google/gemini-2.5-flash-lite",
            "openrouter:qwen/qwen-2.5-7b-instruct",
        ]
        
        all_ok = True
        for test_spec in test_cases:
            if _is_openrouter_spec(test_spec):
                api_base, model = _parse_openrouter_spec(test_spec)
                print(f"✅ {test_spec}")
                print(f"   → API: {api_base}, Model: {model}")
            else:
                print(f"❌ {test_spec} - 解析失败")
                all_ok = False
        
        return all_ok
    except Exception as e:
        print(f"❌ 代码导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def _resolve_config_path(i: int) -> str:
    filename = f"config_agent_openrouter_{i}.yaml"
    candidates = [
        os.path.join("runtime", "configs", "openrouter", filename),
        os.path.join("runtime", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    for root, _dirs, files in os.walk(os.path.join("runtime", "configs", "openrouter")):
        if filename in files:
            return os.path.join(root, filename)
    return candidates[0]


def test_config_files():
    """测试配置文件"""
    print("\n" + "=" * 60)
    print("3. 检查配置文件")
    print("=" * 60)
    
    config_files = []
    for i in range(1, 7):
        config_path = _resolve_config_path(i)
        if os.path.exists(config_path):
            config_files.append(config_path)
            print(f"✅ {config_path} exists")
        else:
            print(f"❌ {config_path} missing")
    
    return len(config_files) == 6

def test_agent_creation():
    """测试创建 Agent"""
    print("\n" + "=" * 60)
    print("4. 测试创建 Agent（不调用 API）")
    print("=" * 60)
    
    try:
        from agents.agent import Agent
        
        config_path = _resolve_config_path(1)
        if not os.path.exists(config_path):
            print(f"❌ 配置文件不存在: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"📋 配置: {config['node_id']}")
        print(f"   模型: {config['base_model']}")
        print(f"   能力: {config.get('capabilities', [])}")
        
        # 注意：这里只是测试配置解析，不实际创建 agent（因为可能需要网络）
        # 如果 base_model 格式正确，配置就应该是正确的
        if config.get('base_model', '').startswith('openrouter:'):
            print("✅ 配置格式正确")
            return True
        else:
            print("❌ 配置格式错误（base_model 应该以 'openrouter:' 开头）")
            return False
            
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("OpenRouter 配置验证")
    print("=" * 60 + "\n")
    
    results = {
        'API Key': test_api_key(),
        '代码支持': test_code_support(),
        '配置文件': test_config_files(),
        'Agent 配置': test_agent_creation(),
    }
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}: {'通过' if passed else '失败'}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！可以开始使用 OpenRouter agents 了。")
        print("\n下一步：")
        print("1. 运行现有实验（exp1, exp3, exp4）- 这些是模拟实验")
        print("2. 或创建新实验使用真实的 OpenRouter agents")
    else:
        print("⚠️  部分测试失败，请检查上述错误信息")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())


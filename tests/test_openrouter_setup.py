#!/usr/bin/env python3
"""
Test OpenRouter configuration is correct
"""
import os
import sys
import yaml

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_api_key():
    """Test if API key is set"""
    print("=" * 60)
    print("1. Check API Key")
    print("=" * 60)
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if api_key:
        print(f"✅ OPENROUTER_API_KEY is set (length: {len(api_key)})")
        print(f"   Prefix: {api_key[:10]}...")
        return True
    else:
        print("❌ OPENROUTER_API_KEY not set")
        print("   Please run: export OPENROUTER_API_KEY='your_key'")
        return False

def test_code_support():
    """Test code support"""
    print("\n" + "=" * 60)
    print("2. Check Code Support")
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
                print(f"❌ {test_spec} - Parse failed")
                all_ok = False
        
        return all_ok
    except Exception as e:
        print(f"❌ Code import failed: {e}")
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
    """Test configuration files"""
    print("\n" + "=" * 60)
    print("3. Check Configuration Files")
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
    """Test Agent creation"""
    print("\n" + "=" * 60)
    print("4. Test Agent Creation (no API call)")
    print("=" * 60)
    
    try:
        from agents.agent import Agent
        
        config_path = _resolve_config_path(1)
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"📋 Config: {config['node_id']}")
        print(f"   Model: {config['base_model']}")
        print(f"   Capabilities: {config.get('capabilities', [])}")
        
        # Note: This only tests config parsing, not actual agent creation (may require network)
        # If base_model format is correct, config should be valid
        if config.get('base_model', '').startswith('openrouter:'):
            print("✅ Config format correct")
            return True
        else:
            print("❌ Config format incorrect (base_model should start with 'openrouter:')")
            return False
            
    except Exception as e:
        print(f"❌ Config load failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("OpenRouter Configuration Validation")
    print("=" * 60 + "\n")
    
    results = {
        'API Key': test_api_key(),
        'Code Support': test_code_support(),
        'Config Files': test_config_files(),
        'Agent Config': test_agent_creation(),
    }
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {test_name}: {'Passed' if passed else 'Failed'}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Ready to use OpenRouter agents.")
        print("\nNext steps:")
        print("1. Run existing experiments (exp1, exp2, exp3) - these are simulation experiments")
        print("2. Or create new experiments using real OpenRouter agents")
    else:
        print("⚠️  Some tests failed, please check error messages above")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())

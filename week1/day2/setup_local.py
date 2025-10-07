import os
import json

def setup_local_project():
    """Quick setup for local prompt engineering project"""
    
    print("🚀 Setting up Local Prompt Engineering Project")
    print("=" * 45)
    
    # Create directories
    directories = ['results', 'logs']
    for dir_name in directories:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✅ Created directory: {dir_name}")
    
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("# Local model setup - no API keys needed\n")
            f.write("LOCAL_MODEL_PATH=./models/\n")
            f.write("LOG_LEVEL=INFO\n")
        print("✅ Created .env file")
    
    # Create config file
    config = {
        "model_settings": {
            "model_name": "Meta-Llama-3-8B-Instruct.Q4_0.gguf",
            "default_temperature": 0.5,
            "default_max_tokens": 200,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.18
        },
        "performance": {
            "enable_logging": True,
            "save_results": True,
            "rate_limiting": 0.5
        }
    }
    
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("✅ Created config.json")
    
    print(f"\n📋 Setup Complete!")
    print(f"Next steps:")
    print(f"1. Make sure GPT4All app has downloaded: Meta-Llama-3-8B-Instruct.Q4_0.gguf")
    print(f"2. Run: python prompt_manager_local.py")
    print(f"3. Test with: python test_local.py")

if __name__ == "__main__":
    setup_local_project()

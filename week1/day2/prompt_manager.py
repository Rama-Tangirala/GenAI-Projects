from gpt4all import GPT4All
import json
import time
import os
from datetime import datetime
import pandas as pd

class PromptTemplate:
    def __init__(self, name, category, system_msg, user_template, description, 
                 examples=None, optimal_temperature=0.7, optimal_max_tokens=200):
        self.name = name
        self.category = category
        self.system_msg = system_msg
        self.user_template = user_template
        self.description = description
        self.examples = examples or []
        self.optimal_temperature = optimal_temperature
        self.optimal_max_tokens = optimal_max_tokens

class PromptManager:
    def __init__(self, model_name="Meta-Llama-3-8B-Instruct.Q4_0.gguf", templates_file="templates.json"):
        print(f"🤖 Initializing Local Prompt Manager with {model_name}")
        
        # Initialize GPT4All model
        try:
            self.model = GPT4All(model_name)
            print(f"✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Make sure the model file is downloaded in GPT4All")
            raise e
        
        self.templates = {}
        self.results_history = []
        self.templates_file = templates_file
        self.load_templates_from_file()
        
        # Create results directory
        if not os.path.exists('results'):
            os.makedirs('results')
    
    def load_templates_from_file(self):
        """Load templates from JSON file"""
        try:
            with open(self.templates_file, 'r') as f:
                data = json.load(f)
                for template_data in data['templates']:
                    template = PromptTemplate(**template_data)
                    self.templates[template.name] = template
            print(f"📚 Loaded {len(self.templates)} templates from {self.templates_file}")
        except FileNotFoundError:
            print(f"📁 Templates file {self.templates_file} not found. Creating default templates.")
            self.create_default_templates_file()
        except Exception as e:
            print(f"❌ Error loading templates: {e}")
    
    def create_default_templates_file(self):
        """Create default templates.json file"""
        default_templates = {
            "templates": [
                {
                    "name": "summarizer_concise",
                    "category": "text_processing",
                    "system_msg": "You are an expert at creating clear, concise summaries. Focus on key points and main ideas.",
                    "user_template": "Please summarize the following text in 2-3 sentences:\n\n{input}",
                    "description": "Creates very concise summaries",
                    "optimal_temperature": 0.3,
                    "optimal_max_tokens": 100
                },
                {
                    "name": "code_explainer",
                    "category": "development",
                    "system_msg": "You are a senior software engineer who explains code clearly to other developers.",
                    "user_template": "Explain this code step by step:\n\n``````",
                    "description": "Explains code for developers",
                    "optimal_temperature": 0.4,
                    "optimal_max_tokens": 200
                },
                {
                    "name": "email_professional",
                    "category": "communication",
                    "system_msg": "You write professional, clear business emails.",
                    "user_template": "Write a professional email for:\n\n{input}",
                    "description": "Professional email writer",
                    "optimal_temperature": 0.3,
                    "optimal_max_tokens": 150
                },
                {
                    "name": "problem_solver",
                    "category": "analysis",
                    "system_msg": "You are a logical problem solver who breaks down problems step-by-step.",
                    "user_template": "Solve this problem step by step:\n\n{input}",
                    "description": "Step-by-step problem solving",
                    "optimal_temperature": 0.2,
                    "optimal_max_tokens": 250
                },
                {
                    "name": "tutor_simple",
                    "category": "education",
                    "system_msg": "You are a patient teacher who explains complex topics simply with examples.",
                    "user_template": "Explain this concept simply with examples:\n\n{input}",
                    "description": "Simple educational explanations",
                    "optimal_temperature": 0.5,
                    "optimal_max_tokens": 200
                }
            ]
        }
        
        with open(self.templates_file, 'w') as f:
            json.dump(default_templates, f, indent=2)
        print(f"✅ Created default templates file: {self.templates_file}")
        self.load_templates_from_file()
    
    def execute_prompt(self, template_name, user_input, temperature=None, max_tokens=None):
        """Execute a specific prompt template with local model"""
        if template_name not in self.templates:
            return {'error': f'Template {template_name} not found'}
        
        template = self.templates[template_name]
        
        # Use optimal parameters if not specified
        temp = temperature if temperature is not None else template.optimal_temperature
        tokens = max_tokens if max_tokens is not None else template.optimal_max_tokens
        
        # Format the complete prompt for local model
        user_prompt = template.user_template.format(input=user_input)
        full_prompt = f"{template.system_msg}\n\n{user_prompt}"
        
        try:
            start_time = time.time()
            
            # Generate with GPT4All
            with self.model.chat_session():
                response = self.model.generate(
                    full_prompt,
                    max_tokens=tokens,
                    temp=temp,
                    top_p=0.9,
                    top_k=40,
                    repeat_penalty=1.18
                )
            
            execution_time = time.time() - start_time
            
            # Estimate token usage (approximate for local models)
            estimated_prompt_tokens = len(full_prompt.split()) * 1.3  # Rough approximation
            estimated_completion_tokens = len(response.split()) * 1.3
            estimated_total_tokens = estimated_prompt_tokens + estimated_completion_tokens
            
            result = {
                'template': template_name,
                'input': user_input,
                'output': response.strip(),
                'estimated_tokens': int(estimated_total_tokens),
                'estimated_prompt_tokens': int(estimated_prompt_tokens),
                'estimated_completion_tokens': int(estimated_completion_tokens),
                'execution_time': execution_time,
                'temperature': temp,
                'max_tokens': tokens,
                'timestamp': datetime.now().isoformat(),
                'success': True,
                'model': "Local Llama-3-8B"
            }
            
            self.results_history.append(result)
            return result
            
        except Exception as e:
            error_result = {
                'template': template_name,
                'input': user_input,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'model': "Local Llama-3-8B"
            }
            self.results_history.append(error_result)
            return error_result
    
    def compare_templates(self, template_names, user_input, save_results=True):
        """Compare multiple templates on the same input"""
        results = {}
        comparison_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n🔬 Running LOCAL comparison: {len(template_names)} templates")
        print(f"Input: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")
        print("-" * 50)
        
        for i, name in enumerate(template_names):
            print(f"Testing template {i+1}/{len(template_names)}: {name}")
            result = self.execute_prompt(name, user_input)
            results[name] = result
            
            if result.get('success'):
                print(f"✅ Success - ~{result['estimated_tokens']} tokens, {result['execution_time']:.2f}s")
                print(f"   Output preview: {result['output'][:100]}...")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
            
            # Small delay to prevent overheating
            time.sleep(0.5)
        
        if save_results:
            self.save_comparison_results(comparison_id, results, user_input)
        
        return results
    
    def save_comparison_results(self, comparison_id, results, input_text):
        """Save comparison results to file"""
        filename = f"results/local_comparison_{comparison_id}.json"
        
        data = {
            'comparison_id': comparison_id,
            'input_text': input_text,
            'model': 'Local Llama-3-8B',
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Results saved to {filename}")
    
    def analyze_performance(self):
        """Analyze performance across all executions"""
        if not self.results_history:
            return {'message': 'No execution history available'}
        
        successful_results = [r for r in self.results_history if r.get('success', False)]
        
        if not successful_results:
            return {'message': 'No successful executions found'}
        
        analysis = {
            'total_executions': len(self.results_history),
            'successful_executions': len(successful_results),
            'success_rate': len(successful_results) / len(self.results_history) * 100,
            'avg_estimated_tokens': sum(r.get('estimated_tokens', 0) for r in successful_results) / len(successful_results),
            'avg_execution_time': sum(r.get('execution_time', 0) for r in successful_results) / len(successful_results),
            'total_estimated_tokens': sum(r.get('estimated_tokens', 0) for r in successful_results),
            'model_used': 'Local Llama-3-8B',
            'template_usage': {},
            'fastest_template': None,
            'most_efficient_template': None
        }
        
        # Template usage statistics
        template_stats = {}
        for result in successful_results:
            template_name = result.get('template', 'unknown')
            if template_name not in template_stats:
                template_stats[template_name] = {
                    'count': 0,
                    'total_time': 0,
                    'total_tokens': 0,
                    'avg_time': 0,
                    'avg_tokens': 0
                }
            
            stats = template_stats[template_name]
            stats['count'] += 1
            stats['total_time'] += result.get('execution_time', 0)
            stats['total_tokens'] += result.get('estimated_tokens', 0)
            stats['avg_time'] = stats['total_time'] / stats['count']
            stats['avg_tokens'] = stats['total_tokens'] / stats['count']
        
        analysis['template_usage'] = template_stats
        
        # Find best performing templates
        if template_stats:
            analysis['fastest_template'] = min(template_stats.keys(), 
                                             key=lambda x: template_stats[x]['avg_time'])
            analysis['most_efficient_template'] = min(template_stats.keys(), 
                                                    key=lambda x: template_stats[x]['avg_tokens'])
        
        return analysis
    
    def interactive_demo(self):
        """Interactive demo of the prompt system"""
        print(f"\n🎮 Interactive Local Prompt Demo")
        print("Available templates:")
        for name, template in self.templates.items():
            print(f"  • {name}: {template.description}")
        
        while True:
            print(f"\n" + "="*40)
            template_name = input("Enter template name (or 'quit'): ").strip()
            
            if template_name.lower() == 'quit':
                break
            
            if template_name not in self.templates:
                print(f"❌ Template '{template_name}' not found")
                continue
            
            user_input = input("Enter your input text: ").strip()
            if not user_input:
                continue
            
            print(f"🤖 Processing with {template_name}...")
            result = self.execute_prompt(template_name, user_input)
            
            if result.get('success'):
                print(f"\n✨ Output:")
                print(result['output'])
                print(f"\n📊 Stats: ~{result['estimated_tokens']} tokens, {result['execution_time']:.2f}s")
            else:
                print(f"❌ Error: {result.get('error')}")

def main():
    """Demo the local prompt manager"""
    try:
        manager = PromptManager()
        
        print("🤖 Local Prompt Engineering System Ready!")
        print("=" * 40)
        
        # List available templates
        print(f"\n📋 Available templates: {len(manager.templates)}")
        for name, template in manager.templates.items():
            print(f"  • {name} ({template.category}): {template.description}")
        
        # Quick demo
        print(f"\n🔍 Quick Demo:")
        result = manager.execute_prompt(
            "summarizer_concise", 
            "Artificial intelligence is transforming industries worldwide. Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions. From healthcare diagnostics to autonomous vehicles, AI applications are becoming increasingly sophisticated and widespread."
        )
        
        if result.get('success'):
            print(f"Template: {result['template']}")
            print(f"Output: {result['output']}")
            print(f"Stats: ~{result['estimated_tokens']} tokens, {result['execution_time']:.2f}s")
        
        return manager
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure GPT4All is installed: pip install gpt4all")
        print("2. Verify model name: 'Meta-Llama-3-8B-Instruct.Q4_0.gguf'")
        print("3. Check if model is downloaded in GPT4All app")
        return None

if __name__ == "__main__":
    manager = main()
    if manager:
        # Run interactive demo
        manager.interactive_demo()

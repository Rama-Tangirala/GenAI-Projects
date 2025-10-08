from gpt4all import GPT4All
import json
import time
from datetime import datetime

class PromptEngineeringLab:
    """Apply your prompt engineering knowledge with local models"""
    
    def __init__(self):
        print("🧪 Prompt Engineering Lab - Applying DeepLearning.AI Concepts")
        self.model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")
        self.experiments = []
        
    def few_shot_prompting(self, task, examples, new_input):
        """Apply few-shot prompting technique from your course"""
        
        # Build few-shot prompt
        prompt = f"Here are some examples of {task}:\n\n"
        
        for i, example in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Input: {example['input']}\n"
            prompt += f"Output: {example['output']}\n\n"
        
        prompt += f"Now do the same for:\nInput: {new_input}\nOutput:"
        
        return self.generate_response(prompt, "few_shot")
    
    def chain_of_thought(self, problem):
        """Apply chain-of-thought prompting"""
        
        prompt = f"""
        Let's think step by step to solve this problem:
        
        Problem: {problem}
        
        Step 1: First, I need to understand what is being asked
        Step 2: Then I'll break it down into smaller parts
        Step 3: I'll work through each part systematically
        Step 4: Finally, I'll provide a clear answer
        
        Let me work through this:
        """
        
        return self.generate_response(prompt, "chain_of_thought")
    
    def role_prompting(self, role, task):
        """Apply role-based prompting technique"""
        
        prompt = f"""
        You are an expert {role}. Based on your expertise and experience, please help with this:
        
        {task}
        
        Please provide your professional perspective:
        """
        
        return self.generate_response(prompt, "role_based")
    
    def iterative_refinement(self, initial_prompt, refinement_notes):
        """Apply iterative prompt refinement"""
        
        results = []
        
        # Try initial prompt
        result1 = self.generate_response(initial_prompt, "iteration_1")
        results.append(result1)
        
        # Refine based on notes
        refined_prompt = f"{initial_prompt}\n\nAdditionally, please {refinement_notes}"
        result2 = self.generate_response(refined_prompt, "iteration_2")
        results.append(result2)
        
        return results
    
    def generate_response(self, prompt, technique):
        """Generate response and track the experiment"""
        
        try:
            start_time = time.time()
            
            with self.model.chat_session():
                response = self.model.generate(
                    prompt,
                    max_tokens=200,
                    temp=0.7
                )
            
            execution_time = time.time() - start_time
            
            experiment = {
                'technique': technique,
                'prompt': prompt,
                'response': response.strip(),
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'prompt_length': len(prompt),
                'response_length': len(response)
            }
            
            self.experiments.append(experiment)
            
            print(f"✅ {technique}: {execution_time:.2f}s")
            return experiment
            
        except Exception as e:
            print(f"❌ Error in {technique}: {e}")
            return None
    
    def run_comprehensive_test(self):
        """Run all prompt engineering techniques on the same problem"""
        
        problem = "How can a small business improve customer satisfaction?"
        
        print(f"\n🎯 Testing Problem: {problem}")
        print("=" * 60)
        
        # 1. Few-shot prompting
        print(f"\n1️⃣ Few-Shot Prompting")
        few_shot_examples = [
            {
                'input': 'How to increase sales?',
                'output': 'Focus on customer needs, improve product quality, enhance marketing'
            },
            {
                'input': 'How to reduce costs?', 
                'output': 'Automate processes, negotiate better supplier rates, eliminate waste'
            }
        ]
        
        few_shot_result = self.few_shot_prompting(
            "business problem solving",
            few_shot_examples, 
            problem
        )
        
        if few_shot_result:
            print(f"Response: {few_shot_result['response']}")
        
        # 2. Chain of thought
        print(f"\n2️⃣ Chain-of-Thought Prompting")
        cot_result = self.chain_of_thought(problem)
        if cot_result:
            print(f"Response: {cot_result['response']}")
        
        # 3. Role prompting
        print(f"\n3️⃣ Role-Based Prompting")
        role_result = self.role_prompting(
            "business consultant with 15 years experience",
            problem
        )
        if role_result:
            print(f"Response: {role_result['response']}")
        
        # 4. Iterative refinement
        print(f"\n4️⃣ Iterative Refinement")
        initial = f"How can a small business improve customer satisfaction?"
        refinement = "focus on actionable, cost-effective strategies that can be implemented within 3 months"
        
        iterations = self.iterative_refinement(initial, refinement)
        
        if iterations:
            print(f"Initial: {iterations[0]['response'][:100]}...")
            print(f"Refined: {iterations[1]['response'][:100]}...")
        
        return self.experiments
    
    def analyze_techniques(self):
        """Analyze which techniques work best"""
        
        if not self.experiments:
            print("No experiments to analyze")
            return
        
        print(f"\n📊 TECHNIQUE ANALYSIS")
        print("=" * 40)
        
        technique_stats = {}
        
        for exp in self.experiments:
            technique = exp['technique']
            if technique not in technique_stats:
                technique_stats[technique] = {
                    'count': 0,
                    'total_time': 0,
                    'total_response_length': 0
                }
            
            stats = technique_stats[technique]
            stats['count'] += 1
            stats['total_time'] += exp['execution_time']
            stats['total_response_length'] += exp['response_length']
        
        # Calculate averages
        for technique, stats in technique_stats.items():
            avg_time = stats['total_time'] / stats['count']
            avg_length = stats['total_response_length'] / stats['count']
            
            print(f"\n{technique.replace('_', ' ').title()}:")
            print(f"  Executions: {stats['count']}")
            print(f"  Avg Time: {avg_time:.2f}s")
            print(f"  Avg Response Length: {avg_length:.0f} chars")
        
    def save_experiments(self):
        """Save all experiments"""
        filename = f"prompt_experiments_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.experiments, f, indent=2)
        
        print(f"\n💾 Saved {len(self.experiments)} experiments to {filename}")
        return filename

def main():
    """Main function"""
    print("🎓 Applying DeepLearning.AI Prompt Engineering Concepts")
    print("=" * 55)
    
    lab = PromptEngineeringLab()
    
    # Run comprehensive test
    experiments = lab.run_comprehensive_test()
    
    # Analyze results
    lab.analyze_techniques()
    
    # Save everything
    lab.save_experiments()
    
    print(f"\n✅ Prompt Engineering Lab Complete!")
    print(f"📚 Applied 4 techniques from your DeepLearning.AI course")
    
    return lab

if __name__ == "__main__":
    lab = main()

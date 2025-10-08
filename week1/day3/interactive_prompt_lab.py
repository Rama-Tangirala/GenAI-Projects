from advanced_patterns import AdvancedPromptPatterns

def interactive_prompt_lab():
    """Interactive lab to test your own prompts"""
    
    print("🧪 Interactive Prompt Engineering Lab")
    print("Apply what you learned from DeepLearning.AI course!")
    print("=" * 50)
    
    lab = AdvancedPromptPatterns()
    
    techniques = {
        '1': ('Few-Shot', 'few_shot_demo'),
        '2': ('Chain-of-Thought', 'chain_of_thought'), 
        '3': ('Role-Based', 'role_demo'),
        '4': ('Delimiter', 'delimiter_prompting'),
        '5': ('Output Format', 'format_demo'),
        '6': ('Temperature Test', 'temperature_comparison'),
        '7': ('Custom Prompt', 'custom')
    }
    
    while True:
        print(f"\n📋 Available Techniques:")
        for key, (name, _) in techniques.items():
            print(f"{key}. {name}")
        print("0. Exit")
        
        choice = input(f"\nChoose technique (0-7): ").strip()
        
        if choice == '0':
            break
        
        if choice not in techniques:
            print("❌ Invalid choice")
            continue
        
        technique_name, method = techniques[choice]
        print(f"\n🎯 Using: {technique_name}")
        
        if choice == '1':  # Few-shot
            task = input("What task? (e.g., 'email writing'): ")
            new_input = input("Your input: ")
            
            examples = [
                {'input': 'Write thank you email', 'output': 'Thank you for your time and consideration...'},
                {'input': 'Write follow-up email', 'output': 'Following up on our previous conversation...'}
            ]
            
            result = lab.few_shot_prompting(task, examples, new_input)
            
        elif choice == '2':  # Chain-of-thought
            problem = input("Enter problem to solve: ")
            result = lab.chain_of_thought(problem)
            
        elif choice == '3':  # Role-based
            role = input("Expert role (e.g., 'software engineer'): ")
            task = input("Task for the expert: ")
            result = lab.role_prompting(role, task)
            
        elif choice == '4':  # Delimiter
            text = input("Enter text to process: ")
            result = lab.delimiter_prompting(text)
            
        elif choice == '5':  # Output format
            question = input("Enter question: ")
            result = lab.output_format_prompting(question)
            
        elif choice == '6':  # Temperature
            prompt = input("Enter prompt for temperature testing: ")
            results = lab.temperature_comparison(prompt)
            print("Results saved to experiments!")
            continue
            
        elif choice == '7':  # Custom
            custom_prompt = input("Enter your custom prompt: ")
            result = lab.generate_response(custom_prompt, "custom")
        
        # Display result
        if result and 'response' in result:
            print(f"\n✨ Response:")
            print(result['response'])
            print(f"⏱️ Time: {result['execution_time']:.2f}s")
        else:
            print("❌ No result generated")
    
    # Save session
    lab.save_experiments()
    lab.analyze_techniques()
    
    print("👋 Lab session ended!")

if __name__ == "__main__":
    interactive_prompt_lab()

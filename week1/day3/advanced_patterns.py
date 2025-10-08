from prompt_engineering_lab import PromptEngineeringLab

class AdvancedPromptPatterns(PromptEngineeringLab):
    """Advanced patterns from your prompt engineering course"""
    
    def delimiter_prompting(self, user_input):
        """Use delimiters to separate instructions from data"""
        
        prompt = f"""
        Please summarize the text delimited by triple backticks in one sentence.
        
        ``````
        
        Summary:
        """
        
        return self.generate_response(prompt, "delimiter_pattern")
    
    def output_format_prompting(self, question):
        """Specify exact output format"""
        
        prompt = f"""
        Answer the following question and format your response as JSON:
        
        Question: {question}
        
        Please provide your response in this exact format:
        {{
            "answer": "your main answer here",
            "confidence": "high/medium/low", 
            "reasoning": "brief explanation of your reasoning"
        }}
        """
        
        return self.generate_response(prompt, "output_format")
    
    def temperature_comparison(self, prompt):
        """Test same prompt with different temperatures (simulated)"""
        
        temperatures = [0.3, 0.7, 1.0]
        results = []
        
        for temp in temperatures:
            print(f"🌡️ Testing temperature {temp}")
            
            # Add temperature instruction to prompt
            temp_prompt = f"""
            [Temperature setting: {temp} - {"Conservative" if temp < 0.5 else "Balanced" if temp < 0.8 else "Creative"}]
            
            {prompt}
            """
            
            result = self.generate_response(temp_prompt, f"temp_{temp}")
            if result:
                result['simulated_temperature'] = temp
                results.append(result)
            
            time.sleep(1)
        
        return results
    
    def run_advanced_tests(self):
        """Run advanced prompt pattern tests"""
        
        print(f"\n🚀 Advanced Prompt Patterns Testing")
        print("=" * 45)
        
        # Test 1: Delimiter prompting
        print(f"\n1️⃣ Delimiter Prompting")
        test_text = """
        Machine learning is a method of data analysis that automates analytical model building. 
        It is a branch of artificial intelligence based on the idea that systems can learn from data, 
        identify patterns and make decisions with minimal human intervention.
        """
        
        delimiter_result = self.delimiter_prompting(test_text)
        if delimiter_result:
            print(f"Result: {delimiter_result['response']}")
        
        # Test 2: Output format prompting  
        print(f"\n2️⃣ Output Format Prompting")
        format_result = self.output_format_prompting(
            "What are the main benefits of using renewable energy?"
        )
        if format_result:
            print(f"Result: {format_result['response']}")
        
        # Test 3: Temperature comparison
        print(f"\n3️⃣ Temperature Comparison")
        temp_results = self.temperature_comparison(
            "Write a creative story about an AI learning to understand emotions."
        )
        
        for result in temp_results:
            temp = result.get('simulated_temperature', 'unknown')
            print(f"\nTemp {temp}: {result['response'][:100]}...")
        
        return [delimiter_result, format_result] + temp_results

def main():
    """Test advanced patterns"""
    
    print("🎓 Advanced Prompt Engineering Patterns")
    print("Based on DeepLearning.AI Course Concepts")
    print("=" * 50)
    
    advanced_lab = AdvancedPromptPatterns()
    
    # Run basic tests first
    print("Running basic techniques...")
    basic_results = advanced_lab.run_comprehensive_test()
    
    # Then advanced patterns
    print("\nRunning advanced patterns...")
    advanced_results = advanced_lab.run_advanced_tests()
    
    # Analysis
    advanced_lab.analyze_techniques()
    advanced_lab.save_experiments()
    
    return advanced_lab

if __name__ == "__main__":
    advanced_lab = main()

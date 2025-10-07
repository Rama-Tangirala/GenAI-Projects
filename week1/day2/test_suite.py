from prompt_manager_local import LocalPromptManager
import time

def run_local_tests():
    """Run comprehensive tests with local model"""
    
    print("🧪 LOCAL MODEL TESTING SUITE")
    print("=" * 40)
    
    try:
        manager = LocalPromptManager()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Test cases optimized for local model performance
    test_cases = [
        {
            "name": "Code Explanation Test",
            "input": """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
            """,
            "templates": ["code_explainer"],
            "expected_keywords": ["recursive", "function", "fibonacci"]
        },
        {
            "name": "Summarization Test", 
            "input": """
Python is a high-level programming language known for its simplicity and readability. 
Created by Guido van Rossum in the late 1980s, Python has become one of the most 
popular languages for web development, data science, artificial intelligence, and 
automation. Its extensive library ecosystem and active community make it an excellent 
choice for both beginners and experienced developers.
            """,
            "templates": ["summarizer_concise"],
            "expected_keywords": ["Python", "programming", "language"]
        },
        {
            "name": "Email Writing Test",
            "input": "Need to reschedule team meeting from Monday to Wednesday due to client presentation",
            "templates": ["email_professional"],
            "expected_keywords": ["meeting", "reschedule", "Wednesday"]
        },
        {
            "name": "Problem Solving Test",
            "input": "How can I improve the performance of a slow database query?",
            "templates": ["problem_solver"],
            "expected_keywords": ["database", "performance", "query"]
        },
        {
            "name": "Educational Test",
            "input": "What is object-oriented programming?",
            "templates": ["tutor_simple"],
            "expected_keywords": ["programming", "objects", "classes"]
        }
    ]
    
    all_results = {}
    total_time = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n🎯 TEST {i}: {test['name']}")
        print(f"Template: {test['templates'][0]}")
        print(f"Input: {test['input'][:100]}{'...' if len(test['input']) > 100 else ''}")
        print("-" * 30)
        
        start_test_time = time.time()
        result = manager.execute_prompt(test['templates'][0], test['input'])
        test_time = time.time() - start_test_time
        total_time += test_time
        
        all_results[test['name']] = result
        
        if result.get('success'):
            output = result['output']
            print(f"✅ Success!")
            print(f"Output: {output}")
            print(f"📊 Metrics: ~{result['estimated_tokens']} tokens, {result['execution_time']:.2f}s")
            
            # Check for expected keywords
            found_keywords = [kw for kw in test['expected_keywords'] 
                            if kw.lower() in output.lower()]
            print(f"🎯 Keywords found: {found_keywords}")
            
            if len(found_keywords) >= 2:
                print(f"✅ Quality check: PASSED")
            else:
                print(f"⚠️  Quality check: PARTIAL")
        else:
            print(f"❌ Error: {result.get('error')}")
        
        print("=" * 50)
        time.sleep(1)  # Prevent overheating
    
    # Performance summary
    print(f"\n📈 PERFORMANCE SUMMARY")
    print(f"Total test time: {total_time:.2f}s")
    print(f"Average per test: {total_time/len(test_cases):.2f}s")
    
    analysis = manager.analyze_performance()
    if 'message' not in analysis:
        print(f"Success rate: {analysis['success_rate']:.1f}%")
        print(f"Average tokens per request: {analysis['avg_estimated_tokens']:.1f}")
        print(f"Average execution time: {analysis['avg_execution_time']:.2f}s")
        print(f"Total estimated tokens: {analysis['total_estimated_tokens']}")
        
        if analysis['fastest_template']:
            print(f"🏃‍♂️ Fastest template: {analysis['fastest_template']}")
        if analysis['most_efficient_template']:
            print(f"💡 Most efficient template: {analysis['most_efficient_template']}")
    
    return manager, all_results

if __name__ == "__main__":
    manager, results = run_local_tests()
    
    if manager:
        print(f"\n🎉 Testing complete!")
        print(f"💡 Pro tip: Your local model has no API costs - experiment freely!")
        
        # Offer interactive mode
        interactive = input("\nWould you like to try interactive mode? (y/n): ")
        if interactive.lower() == 'y':
            manager.interactive_demo()

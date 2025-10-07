import argparse
from transformers import pipeline
import torch

class FreeAITextCompleter:
    def __init__(self, model_name='gpt2-medium', temperature=0.7):
        print(f"Loading model: {model_name}")
        self.generator = pipeline(
            'text-generation',
            model=model_name,
            device=0 if torch.cuda.is_available() else -1  # GPU if available, else CPU
        )
        self.temperature = temperature
    
    def complete_text(self, prompt, max_tokens=100, temperature=None):
        temp = temperature if temperature is not None else self.temperature
        try:
            result = self.generator(
            prompt,
            max_new_tokens=max_tokens,    # number of tokens to generate
            temperature=temp,
            do_sample=True,
            pad_token_id=self.generator.tokenizer.eos_token_id,
            truncation=True,
            num_return_sequences=1
            )
            return result[0]['generated_text']
        except Exception as e:
            return f"Error generating text: {str(e)}"
    
    def interactive_mode(self):
        print("🤖 Free AI Text Completer Ready!")
        print("Type 'quit' or 'exit' to stop\n")
        while True:
            try:
                prompt = input("You: ").strip()
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                if not prompt:
                    continue
                print("AI: ", end="", flush=True)
                response = self.complete_text(prompt)
                new_text = response[len(prompt):].strip()
                print(new_text)
                print("-" * 50)
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break

def main():
    parser = argparse.ArgumentParser(description='Free AI Text Completion Tool')
    parser.add_argument('--prompt', type=str, help='Text prompt to complete')
    parser.add_argument('--model', type=str, default='gpt2-medium',
                        help='HuggingFace model name (default: gpt2-medium)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Generation temperature (0.1-2.0)')
    parser.add_argument('--max-tokens', type=int, default=100,
                        help='Maximum tokens to generate')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')

    args = parser.parse_args()

    completer = FreeAITextCompleter(
        model_name=args.model,
        temperature=args.temperature
    )

    if args.interactive:
        completer.interactive_mode()
    elif args.prompt:
        result = completer.complete_text(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        print("Generated text:")
        print(result)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

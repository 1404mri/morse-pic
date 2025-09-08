import random
import argparse
import os
from typing import List, Dict, Any
from google import  genai
from google.genai import types
from io import BytesIO

def load_mathvista_dataset(dataset_name: str = "AI4Math/MathVista") -> List[Dict[str, Any]]:
    """Load MathVista dataset from Hugging Face or return sample data if loading fails."""
    from datasets import load_dataset
    dataset = load_dataset(dataset_name)["testmini"]
    return [dict(item) for item in dataset]

def select_examples(dataset: List[Dict[str, Any]], num_examples: int = 3) -> List[Dict[str, Any]]:
    """Randomly select examples from the dataset."""
    return dataset if len(dataset) <= num_examples else random.sample(dataset, num_examples)

def create_gemini_prompt(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create content for Gemini API with examples and request for harder problems."""
    contents = [
        "You are an expert math educator. I'm working on creating more challenging versions of math problems for advanced students. Here are some example problems:\n\n"
    ]
     
    for i, example in enumerate(examples, 1):
        contents.append(f"Example {i}:\n")
        img = example['decoded_image']
        buffer = BytesIO()
        img.save(buffer, format=img.format)
        contents.extend([
            types.Part.from_bytes(
                data=buffer.getvalue(), 
                mime_type=f'image/{img.format.lower()}'
            ), 
            f"\nQuestion: {example['question']}\n", 
            f"Answer: {example['answer']}\n",
            f"Grade Level: {example['metadata']['grade']}\n",
            f"Question Type: {example.get('question_type', 'N/A')}\n\n"
        ])
    
    contents.append(
    """Please provide ideas for making these problems more challenging. Consider:
    - Adding multi-step reasoning\n
    - Incorporating additional mathematical concepts\n
    - Increasing abstraction or complexity\n
    - Requiring deeper conceptual understanding\n
    - Introducing real-world applications\n\n
For each example, suggest 2-3 specific ways to increase the difficulty while maintaining the core mathematical concept.
    """)
    return contents

def call_gemini_api(contents) -> str:
    """Call the Gemini API with prompt and images."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    client = genai.Client()

    # Initialize model and generate content
    response = client.models.generate_content(
                model = "gemini-2.5-pro", 
                contents=contents
               )
    return response.text


def main():
    parser = argparse.ArgumentParser(description="Generate prompts for Gemini to suggest ways to make math problems more challenging.")
    parser.add_argument("--dataset", type=str, default="AI4Math/MathVista", help="Hugging Face dataset name (default: AI4Math/MathVista)")
    parser.add_argument("--num_examples", type=int, default=3, help="Number of examples (default: 3)")
    parser.add_argument("--output", type=str, default="gemini_prompt.txt", help="Output file path (default: gemini_prompt.txt)")
    parser.add_argument("--call-api", action="store_true", help="Call the Gemini API with the generated prompt and images")
    
    args = parser.parse_args()
    
    try:  
        dataset = load_mathvista_dataset(args.dataset)
        print("Successfully loaded MathVista")
    except:
        print("MathVista couldn't get loaded.")
        return

    examples = select_examples(dataset, args.num_examples)
    contents = create_gemini_prompt(examples)
    
    with open(args.output, "w") as f:
        f.write("".join([str(c) if type(c) != types.Part else "<img>" for c in contents]))
    print("Saved prompt to ", args.output) 
    
    if args.call_api:
            
        try:
            print(f"\nCalling Gemini API ....")
            response = call_gemini_api(contents)
            
            print("Gemini API Response:")
            print("=" * 50)
            print(response)
            print("=" * 50)
            
            # Save response to file
            with open("gemini_response.txt", "w") as f:
                f.write(response)
            print("\nResponse saved to 'gemini_response.txt'")
            
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return 1

if __name__ == "__main__":
    main()

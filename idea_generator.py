import random
import argparse
import os
from typing import List, Dict, Any
from google import genai
from google.genai import types
from io import BytesIO

BATCH_SIZE = 100  # Number of problems to process in each batch

def load_mathvista_dataset(dataset_name: str = "AI4Math/MathVista") -> List[Dict[str, Any]]:
    """Load MathVista dataset from Hugging Face or return sample data if loading fails."""
    from datasets import load_dataset
    dataset = load_dataset(dataset_name)["testmini"]
    return [dict(item) for item in dataset]

def load_context_examples(file_path: str = "context_examples/context_examples.json") -> List[Dict[str, Any]]:
    """Load examples from the context_examples.json file."""
    import json
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def select_examples(dataset: List[Dict[str, Any]], num_examples: int = 3) -> List[Dict[str, Any]]:
    """Randomly select examples from the dataset."""
    return dataset if len(dataset) <= num_examples else random.sample(dataset, num_examples)

def create_gemini_prompt(dataset: List[Dict[str, Any]], examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create content for Gemini API with examples and request for harder problems."""
    contents = [
        "You are an expert math educator. Given a list of problems, you need to list a way of controlling the difficulty level of each problem so that a coding assistant can generate code to do along with the list of original questions. Here's a sample expected output: \n\n"
    ]
     
    for i, example in enumerate(examples, 1):
        # Load the image from the provided path
        import os
        from PIL import Image
        
        # The images are in the context_examples/images subdirectory relative to the script
        image_path = os.path.join(os.path.dirname(__file__), 'context_examples', example['Image Path'])
        img = Image.open(image_path)
        buffer = BytesIO()
        img.save(buffer, format='PNG')  # Save as PNG format
        
        contents.extend([
            "**Example**\n",
            types.Part.from_bytes(
                data=buffer.getvalue(), 
                mime_type='image/png'
            ), 
            f"\nQuestion: {example['Question']}\n", 
            f"Answer/Solution: {example['Solution']}\n",
            f"Image Description: {example['Image Description']}\n",
            f"Difficulty Control: {example['Difficulty Control']}\n\n"
        ])
    
    contents.append("Now, for each of these problems, list the problem number and a proposed difficulty control technique for it similar to the ones above:\n")
    p_no = 0
    for i, example in enumerate(dataset, 1):
        img = example['decoded_image']
        buffer = BytesIO()
        img.save(buffer, format=img.format)
        if img.format.lower() not in ["png", "jpeg", "webp", "heic", "heif"]: # gemini support types
            print(f"Skipping example {i} because it's img format, {img.format.lower()}, isn't supported.")
            continue
        p_no += 1
        contents.append(f"**Problem {p_no}**\n")
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
    parser.add_argument("--json-file", type=str, default="context_examples/context_examples.json", help="JSON file containing examples (default: context_examples/context_examples.json)")
    parser.add_argument("--num_examples", type=int, default=3, help="Number of in-context examples (default: 3)")
    parser.add_argument("--output", type=str, default="gemini_prompt.txt", help="Output file path (default: gemini_prompt.txt)")
    parser.add_argument("--call-api", action="store_true", help="Call the Gemini API with the generated prompt and images")
    
    args = parser.parse_args()

    try:
        dataset = load_mathvista_dataset()
        print("Successfully loaded MathVista")
    except Exception as e:
        print("MathVista couldn't get loaded because of this error:")
        print(e)
        return
    
    try:  
        examples = load_context_examples(args.json_file)
        print("Successfully loaded context examples")
    except FileNotFoundError:
        print(f"Context examples file {args.json_file} couldn't be found.")
        return
    except Exception as e:
        print(f"Error loading context examples: {e}")
        return

    for i in range(0, len(dataset), BATCH_SIZE):
        dataset_batch = dataset[i:i + BATCH_SIZE]
        contents = create_gemini_prompt(dataset_batch, examples)
        
        with open(args.output.removesuffix(".txt") + f"_b{i // 100}.txt", "w") as f:
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
                with open(f"gemini_response_b{i // 100}.txt", "w") as f:
                    f.write(response)
                print(f"\nResponse saved to 'gemini_response_b{i // 100}.txt'")
                
            except Exception as e:
                print(f"Error calling Gemini API: {e}")
                return 1

if __name__ == "__main__":
    main()

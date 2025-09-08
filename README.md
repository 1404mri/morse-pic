# morse500-v2

### Files

- `idea_generator.py`: Main script that loads the dataset and generates prompts
- `requirements.txt`: Python dependencies (requires `datasets` library)

### Installation

Before running the script, install the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

```bash
python idea_generator.py
```

The script will:
1. Load the MathVista dataset from Hugging Face (uses sample data if loading fails)
2. Select a few random examples from the dataset
3. Generate a prompt for Gemini 2.5 Pro asking for ideas to make the problems harder
4. Save the prompt to `gemini_prompt.txt`

### Command-line Options

- `--dataset NAME`: Name of the Hugging Face dataset (default: AI4Math/MathVista)
- `--num_examples N`: Number of examples to include in the prompt (default: 3)
- `--output PATH`: Output file path (default: gemini_prompt.txt)

### Examples

```bash
# Basic usage with default Hugging Face dataset
python idea_generator.py

# With custom dataset name
python idea_generator.py --dataset AI4Math/MathVista

# With custom number of examples and output file
python idea_generator.py --dataset AI4Math/MathVista --num_examples 5 --output harder_problems_prompt.txt
```

### Output

The generated prompt will be printed to the console and saved to the specified output file.

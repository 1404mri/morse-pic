# MORSE-PIC: Mathematical Visual Question Generation System

A DSPy-based system for generating programmatic mathematical visualizations with corresponding questions and ground truth answers. This system uses Vision Language Models (VLMs) to generate Python code that creates mathematical images and calculates answers programmatically.

## 🚀 Features

- **VLM-Powered Code Generation**: Uses DSPy signatures to generate complete Python visualization code
- **Programmatic Ground Truth**: Calculates answers mathematically, not through AI inference
- **Difficulty Scaling**: Unlimited difficulty levels with automatic complexity adjustments
- **Multiple Image Contexts**: Supports 14 different visualization types
- **Multi-Model Support**: Works with OpenAI, Ollama, and Gemini models
- **Automated Verification**: Built-in code verification and mathematical accuracy checking
- **Batch Processing**: Generate multiple visualizations efficiently

## 📦 Installation

### Prerequisites
```bash
# Install Python 3.8+
python --version

# Install DSPy
pip install dspy-ai

# Install visualization libraries
pip install matplotlib seaborn pandas numpy sympy plotly pillow
```

### Optional Dependencies
```bash
# For advanced 3D visualizations
pip install pyvista trimesh open3d

# For geographic visualizations
pip install folium

# For computer vision
pip install opencv-python

# For Ollama support
pip install ollama
```

### Clone Repository
```bash
git clone https://github.com/1404mri/MORSE-PIC.git
cd MORSE-PIC
pip install -r requirements.txt
```

## 🎯 Quick Start

### Basic Usage

```python
from coding_agent import MathVistaSystem, GenerationInput

# Initialize the system
system = MathVistaSystem()

# Create input specification
input_data = GenerationInput(
    initial_question="What is the area of the triangle?",
    reference_image_path=None,
    image_description="A right triangle with sides 3, 4, and 5",
    difficulty_control="Add angle measurements and multiple triangles",
    difficulty_level=3,
    image_context="geometry_diagram"
)

# Generate visualization
output = system.generate(input_data)

print(f"Generated Question: {output.question}")
print(f"Image Path: {output.image_path}")
print(f"Ground Truth: {output.ground_truth}")
print(f"Generated Code:\n{output.generated_code}")
```

### Model Configuration

#### OpenAI (Default)
```python
system = MathVistaSystem(model_name="gpt-4")
```

#### Ollama
```python
system = MathVistaSystem()
system.configure_for_ollama(
    base_url="http://localhost:11434", 
    model="codellama"
)
```

#### Gemini
```python
system = MathVistaSystem()
system.configure_for_gemini(
    api_key="your-gemini-api-key",
    model="gemini-pro"
)
```

## 📊 Supported Image Contexts

| Context | Description | Libraries Used |
|---------|-------------|----------------|
| `bar_chart` | Bar charts with statistical analysis | matplotlib, seaborn, pandas |
| `function_plot` | Mathematical function visualizations | matplotlib, numpy, sympy |
| `geometry_diagram` | Geometric shapes and calculations | matplotlib, numpy |
| `scatter_plot` | Correlation and regression analysis | matplotlib, seaborn, numpy |
| `line_plot` | Time series and trend analysis | matplotlib, pandas |
| `pie_chart` | Categorical data distribution | matplotlib |
| `violin_plot` | Statistical distribution visualization | seaborn, matplotlib |
| `table` | Structured data with calculations | pandas, matplotlib |
| `scientific_figure` | Scientific data visualization | matplotlib, plotly |
| `abstract_scene` | Abstract mathematical concepts | matplotlib, turtle |
| `document_image` | Mathematical documents/equations | PIL, matplotlib, sympy |
| `puzzle_test` | Visual mathematical puzzles | PIL, matplotlib |
| `synthetic_scene` | 3D synthetic environments | matplotlib, pyvista |
| `map_chart` | Geographic/spatial visualizations | folium, plotly |

## 🎛️ Difficulty Control

The system supports unlimited difficulty levels with automatic scaling:

- **Level 1-2**: Basic visualizations with simple calculations
- **Level 3-4**: Intermediate complexity with additional elements
- **Level 5+**: Advanced features with complex mathematical operations

### Difficulty Parameters
```python
# Automatic scaling based on difficulty level
params = {
    "num_data_points": 10 + 5 * difficulty,
    "decimal_places": min(difficulty, 4),
    "num_categories": min(3 + difficulty, 15),
    "calculation_steps": difficulty,
    "visual_elements": min(difficulty * 2, 20)
}
```

## 🔧 Advanced Usage

### Batch Processing
```python
batch_inputs = [
    GenerationInput(
        initial_question="What is the correlation?",
        image_description="Scatter plot with positive correlation",
        difficulty_control="Add regression analysis",
        difficulty_level=4,
        image_context="scatter_plot"
    ),
    GenerationInput(
        initial_question="Which category is largest?",
        image_description="Bar chart comparison",
        difficulty_control="Add percentage labels",
        difficulty_level=2,
        image_context="bar_chart"
    )
]

results = system.batch_generate(batch_inputs)
```

### Quality Evaluation
```python
quality_scores = system.evaluate_generation_quality(output)
print(f"Overall Quality: {quality_scores['overall']:.2f}")
print(f"Code Executability: {quality_scores['code_executability']}")
print(f"Ground Truth Validity: {quality_scores['ground_truth_validity']}")
```

### Pipeline Optimization
```python
# Optimize DSPy pipeline with training examples
training_data = [(input1, output1), (input2, output2), ...]
system.optimize_pipeline(training_data)
```

## 📋 Input Specification

### GenerationInput Fields

| Field | Type | Description |
|-------|------|-------------|
| `initial_question` | str | Base mathematical question |
| `reference_image_path` | Optional[str] | Path to reference image (can be None) |
| `image_description` | str | Detailed description of desired image |
| `difficulty_control` | str | Instructions for adjusting complexity |
| `difficulty_level` | int | Numerical difficulty (1 to ∞) |
| `image_context` | str | Type of visualization to generate |

### Example Inputs by Context

#### Geometry Diagram
```python
GenerationInput(
    initial_question="Find the area of the triangle",
    reference_image_path=None,
    image_description="Right triangle with labeled sides",
    difficulty_control="Add angle bisectors and altitude measurements",
    difficulty_level=5,
    image_context="geometry_diagram"
)
```

#### Function Plot
```python
GenerationInput(
    initial_question="What is the maximum value of the function?",
    reference_image_path=None,
    image_description="Quadratic function with vertex form",
    difficulty_control="Add derivative and critical points analysis",
    difficulty_level=6,
    image_context="function_plot"
)
```

#### Statistical Visualization
```python
GenerationInput(
    initial_question="What is the correlation coefficient?",
    reference_image_path=None,
    image_description="Scatter plot with strong positive correlation",
    difficulty_control="Add confidence intervals and R-squared value",
    difficulty_level=4,
    image_context="scatter_plot"
)
```

## 📊 Output Specification

### GenerationOutput Fields

| Field | Type | Description |
|-------|------|-------------|
| `question` | str | Adapted question matching the generated image |
| `image_path` | str | Path to the generated image file |
| `ground_truth` | str | Programmatically calculated answer |
| `metadata` | Dict[str, Any] | Generation metadata and parameters |
| `generated_code` | str | Complete Python code that created the image |

### Example Output
```python
output = GenerationOutput(
    question="What is the area of triangle ABC with sides 5, 12, and 13?",
    image_path="generated_images/geometry_diagram_3.png",
    ground_truth="{'area': 30.0, 'perimeter': 30, 'type': 'right_triangle'}",
    metadata={
        "difficulty_level": 3,
        "image_context": "geometry_diagram",
        "recommended_libraries": ["matplotlib", "numpy"],
        "verification_result": {...}
    },
    generated_code="import matplotlib.pyplot as plt\nimport numpy as np\n..."
)
```

## 🔍 System Architecture

### Core Components

1. **CodingAgent**: Main DSPy module for code generation
2. **VerifierAgent**: Code verification and mathematical accuracy checking
3. **CodeExecutor**: Safe code execution environment
4. **LibrarySelector**: Context-aware library recommendation
5. **DifficultyController**: Automatic complexity scaling

### DSPy Signatures

- **QuestionAnalysis**: Analyzes input questions and requirements
- **CodeGeneration**: Generates complete Python visualization code
- **CodeVerification**: Validates generated code quality
- **MathematicalAccuracy**: Verifies mathematical correctness

## 🐛 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Install missing packages
pip install matplotlib seaborn pandas numpy sympy
```

#### DSPy Configuration
```python
# Ensure proper model configuration
import dspy
dspy.settings.configure(lm=dspy.OpenAI(model="gpt-3.5-turbo"))
```

#### Code Execution Failures
- Check that all required libraries are installed
- Verify the generated code syntax
- Ensure output directory exists and is writable

### Error Handling
The system includes automatic retry mechanisms:
- Up to 3 retries for failed code generation
- Automatic package installation
- Comprehensive error reporting

## 📈 Performance Tips

1. **Model Selection**: Use `gpt-4` or `codellama` for better code generation
2. **Library Installation**: Pre-install all visualization libraries
3. **Batch Processing**: Use batch generation for multiple items
4. **Difficulty Scaling**: Start with lower difficulty levels for testing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 🙋‍♀️ Support

- Create an issue on GitHub for bugs
- Check existing issues for common problems
- Review the troubleshooting section above

## 📚 Examples

See the `examples/` directory for:
- Complete usage examples
- Different visualization types
- Advanced configuration options
- Integration patterns

## 🔗 Related Work

- [DSPy Framework](https://github.com/stanfordnlp/dspy)
- [MathVista Dataset](https://mathvista.github.io/)
- [Mathematical Reasoning with Vision](https://arxiv.org/abs/2310.02255)

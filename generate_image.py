"""
Standalone script that reuses the CodingAgent pipeline to generate visualization code
and save the raw Python output to disk. This does not import from coding_agent.py;
all necessary components are defined inline here.
"""

import os
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import re
import dspy
import dotenv


dotenv.load_dotenv()

def extract_python_code(generated_code: str) -> str:
    """Extract the Python code block from the generated code string."""
    code_block_pattern = re.compile(r"```python(.*?)```", re.DOTALL)
    match = code_block_pattern.search(generated_code)
    if match:
        return match.group(1).strip()
    else:
        # If no code block is found, return the original string
        return generated_code.strip()


class ImageContext(Enum):
	ABSTRACT_SCENE = "abstract scene"
	BAR_CHART = "bar chart"
	DOCUMENT_IMAGE = "document image"
	FUNCTION_PLOT = "function plot"
	GEOMETRY_DIAGRAM = "geometry diagram"
	LINE_PLOT = "line plot"
	MAP_CHART = "map chart"
	PIE_CHART = "pie chart"
	PUZZLE_TEST = "puzzle test"
	SCATTER_PLOT = "scatter plot"
	SCIENTIFIC_FIGURE = "scientific figure"
	SYNTHETIC_SCENE = "synthetic scene"
	TABLE = "table"
	VIOLIN_PLOT = "violin plot"


@dataclass
class GenerationInput:
	initial_question: str
	reference_image_path: Optional[str]
	image_description: str
	difficulty_control: str
	difficulty_level: int
	image_context: str


class LibrarySelector:
	CONTEXT_LIBRARY_MAP = {
		ImageContext.ABSTRACT_SCENE: ["matplotlib", "turtle", "PIL", "numpy"],
		ImageContext.BAR_CHART: ["matplotlib", "seaborn", "plotly", "pandas"],
		ImageContext.DOCUMENT_IMAGE: ["PIL", "matplotlib", "sympy", "cv2"],
		ImageContext.FUNCTION_PLOT: ["matplotlib", "numpy", "sympy", "plotly"],
		ImageContext.GEOMETRY_DIAGRAM: ["matplotlib", "turtle", "PIL", "numpy"],
		ImageContext.LINE_PLOT: ["matplotlib", "seaborn", "plotly", "pandas"],
		ImageContext.MAP_CHART: ["folium", "plotly", "matplotlib", "pandas"],
		ImageContext.PIE_CHART: ["matplotlib", "plotly", "seaborn"],
		ImageContext.PUZZLE_TEST: ["PIL", "matplotlib", "turtle", "numpy"],
		ImageContext.SCATTER_PLOT: ["matplotlib", "seaborn", "plotly", "pandas"],
		ImageContext.SCIENTIFIC_FIGURE: ["matplotlib", "plotly", "numpy", "seaborn"],
		ImageContext.SYNTHETIC_SCENE: ["blender", "pyvista", "trimesh"], #, "open3d"
		ImageContext.TABLE: ["pandas", "matplotlib", "PIL", "plotly"],
		ImageContext.VIOLIN_PLOT: ["seaborn", "matplotlib", "plotly", "pandas"],
	}

	@classmethod
	def get_libraries(cls, context: str) -> List[str]:
		try:
			context_enum = ImageContext(context.lower())
			return cls.CONTEXT_LIBRARY_MAP.get(context_enum, ["matplotlib", "numpy", "PIL"])
		except ValueError:
			return ["matplotlib", "numpy", "PIL"]


class DifficultyController:
    """Controls difficulty scaling for questions and images"""
    
    @staticmethod
    def scale_numerical_parameters(base_value: float, difficulty: int, scaling_type: str = "linear") -> float:
        """Scale numerical parameters based on difficulty level"""
        if scaling_type == "linear":
            return base_value * (1 + 0.2 * (difficulty - 1))
        elif scaling_type == "exponential":
            return base_value * (1.5 ** (difficulty - 1))
        elif scaling_type == "logarithmic":
            import math
            return base_value * (1 + 0.3 * math.log(difficulty))
        else:
            return base_value
    
    @staticmethod
    def get_complexity_parameters(difficulty: int) -> Dict[str, Any]:
        """Get complexity parameters based on difficulty level"""
        params = {
            "num_data_points": min(10 + 5 * difficulty, 100),
            "decimal_places": min(difficulty, 4),
            "num_categories": min(3 + difficulty, 15),
            "calculation_steps": difficulty,
            "visual_elements": min(difficulty * 2, 20),
            "complexity_level": "basic" if difficulty <= 2 else "intermediate" if difficulty <= 4 else "advanced"
        }
        return params


class QuestionAnalysis(dspy.Signature):
    """Analyze the initial question and determine adaptation requirements"""
    initial_question = dspy.InputField()
    difficulty_control = dspy.InputField()
    difficulty_level = dspy.InputField()
    image_context = dspy.InputField()
    
    analysis_result = dspy.OutputField(desc="Detailed analysis of question requirements and adaptation strategy")
    mathematical_concepts = dspy.OutputField(desc="Key mathematical concepts identified in the question")
    adaptation_strategy = dspy.OutputField(desc="Strategy for adapting question to new difficulty level")


class CodeGeneration(dspy.Signature):
    """Generate Python code for mathematical visualization with programmatic ground truth calculation. The code must be complete, executable, and save the image to 'output_path'. Required variables: 'ground_truth' (dict) and 'final_image_path' (str). Use only the recommended libraries and scale complexity based on difficulty level. your code should have a variable named difficulty_level"""
    
    question_analysis = dspy.InputField(desc="Analysis of the original question and adaptation requirements")
    mathematical_concepts = dspy.InputField(desc="Key mathematical concepts to implement in the visualization")
    recommended_libraries = dspy.InputField(desc="Specific libraries to use")
    difficulty_parameters = dspy.InputField(desc="Complexity parameters including num_data_points, decimal_places, visual_elements")
    image_context = dspy.InputField(desc="Type of visualization (bar_chart, function_plot, geometry_diagram, scatter_plot, etc.)")
    initial_question = dspy.InputField(desc="Original mathematical question to base the visualization on")
    difficulty_control = dspy.InputField(desc="Instructions for modifying difficulty level")
    
    generated_code = dspy.OutputField(desc="Complete executable Python code that: 1) Imports required libraries, 2) Generates/creates data programmatically, 3) Creates the visualization, 4) Calculates ground truth mathematically (not using AI), 5) Saves image to output_path, 6) Sets final_image_path=output_path, 7) Sets ground_truth as dictionary with calculated values")
    adapted_question = dspy.OutputField(desc="Mathematical question adapted to match the generated visualization and target difficulty level")
    code_explanation = dspy.OutputField(desc="Clear explanation of what the code does, how ground truth is calculated, and what mathematical concepts are demonstrated")


class CodingAgent(dspy.Module):
    """Main coding agent module for generating mathematical visualizations via DSPy signatures"""
    
    def __init__(self, enable_thinking: bool = True):
        super().__init__()
        # Use ChainOfThought only if thinking is enabled (not for thinking models)
        if enable_thinking:
            self.question_analyzer = dspy.ChainOfThought(QuestionAnalysis)
            self.code_generator = dspy.ChainOfThought(CodeGeneration)
        else:
            # For thinking models, use direct signatures without ChainOfThought wrapper
            self.question_analyzer = dspy.Predict(QuestionAnalysis)
            self.code_generator = dspy.Predict(CodeGeneration)
        
        self.library_selector = LibrarySelector()
        self.difficulty_controller = DifficultyController()
    
    def forward(self, generation_input: GenerationInput) -> Dict[str, Any]:
        """Forward pass of the coding agent using DSPy signatures for code generation"""
        
        # Step 1: Analyze the question
        analysis = self.question_analyzer(
            initial_question=generation_input.initial_question,
            difficulty_control=generation_input.difficulty_control,
            difficulty_level=str(generation_input.difficulty_level),
            image_context=generation_input.image_context
        )
        
        # Step 2: Select appropriate libraries
        recommended_libraries = self.library_selector.get_libraries(generation_input.image_context)
        
        # Step 3: Get difficulty parameters
        difficulty_params = self.difficulty_controller.get_complexity_parameters(generation_input.difficulty_level)
        
        # Step 4: Use DSPy signature to generate code (no explicit prompting)
        code_result = self.code_generator(
            question_analysis=analysis.analysis_result,
            mathematical_concepts=analysis.mathematical_concepts,
            recommended_libraries=str(recommended_libraries),
            difficulty_parameters=str(difficulty_params),
            image_context=generation_input.image_context,
            initial_question=generation_input.initial_question,
            difficulty_control=generation_input.difficulty_control
        )
        
        return {
            "analysis": analysis,
            "code_result": code_result,
            "recommended_libraries": recommended_libraries,
            "difficulty_params": difficulty_params
        }


def setup_model(model_type: str = "gemini", model_name: str = "gemini-1.5-flash") -> bool:
	api_key = os.getenv("GEMINI_API_KEY")
	thinking_models = ["qwen", "qwen2", "qwen3", "deepseek", "o1"]

	if model_type.lower() == "ollama":
		lm = dspy.LM(
			f"ollama_chat/{model_name}",
			api_base="http://localhost:11434",
			api_key="",
			max_tokens=32000,
		)
		has_built_in_thinking = any(
			identifier in model_name.lower() for identifier in thinking_models
		)
	elif model_type.lower() == "gemini":
		if not api_key:
			raise ValueError(
				"GEMINI_API_KEY must be set in your environment or .env file"
			)
		lm = dspy.LM(f"gemini/{model_name}", api_key=api_key, max_tokens=32000)
		has_built_in_thinking = False
	else:
		raise ValueError("Supported model types are 'gemini' and 'ollama'.")

	dspy.configure(lm=lm)
	return not has_built_in_thinking


def save_generated_code(code: str, metadata: Dict[str, Any]) -> Path:
    output_dir = Path("generated_code")
    output_dir.mkdir(exist_ok=True)
    code = extract_python_code(code)
    safe_context = metadata.get("image_context", "visualization").replace(" ", "_")
    filename = f"generated_{safe_context}.py"
    output_path = output_dir / filename

    metadata_path = output_dir / "metadata.json"

    payload = {
        "metadata": metadata,
        "generated_code": code,
    }

    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Auto-generated visualization code\n")
        f.write(code)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"💾 Code saved to {output_path}")
    print(f"💾 Metadata saved to {metadata_path}")
    return output_path


def main() -> None:
	model_type = os.getenv("MODEL_TYPE", "gemini")
	model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")

	enable_thinking = setup_model(model_type, model_name)
	agent = CodingAgent(enable_thinking=enable_thinking)

	generation_input = GenerationInput(
		initial_question="""Is the number of tiny gray bicycles that are on the left side of the brown metal sedan greater than the number of things that are to the left of the tiny green bicycle?""",
		reference_image_path="img/image_1.png",
		image_description="""
        The image is a 3D rendering of several vehicles on a plain, light-colored surface. The lighting is from the upper left, casting subtle shadows. The vehicles are: 
        A metallic red motorcycle: Highly stylized with a streamlined, almost futuristic design. It has intricate, chrome-like details, and a light-blue or cyan seat and wheel rims. It is positioned near the center-left of the image.
        A metallic gold sedan: A large, four-door car with a sleek, reflective gold finish. It's positioned on the right side of the image, facing slightly left.
        A small, pink sedan: A small, simple-looking car with a pink body and a light-blue roof. It's located behind the red motorcycle and the green bicycle.
        A green bicycle: A classic-style bicycle with a bright green frame and a thin, turquoise saddle and wheels. It is situated to the left of the gray dirt bike.
        A gray dirt bike: A large, uncolored or matte gray dirt bike with a green front fender and blue wheels. It's the largest vehicle in the group and is positioned in the center of the image.
        Two joined bicycles: A unique, low-profile vehicle consisting of two bicycle-like frames joined together side-by-side, lying on the ground. One side is blue and the other is a metallic orange or bronze color. This vehicle is in the foreground, at the bottom of the image.
        The overall scene has a somewhat surreal or artistic feel due to the mix of vehicle styles and the varied, often reflective, materials.       
        """,
		difficulty_control="""
		To adjust the difficulty for a question about this image, consider manipulating the context and relationships between the objects. For instance, to increase the difficulty, you could introduce a narrative or a physical interaction: "Imagine these vehicles are part of a race, and the two joined bicycles are obstacles. The red motorcycle needs to choose the shortest path around them to reach the finish line, which is marked by the golden car. Calculate the minimal distance if each unit represents one meter, and the vehicles are currently at specific 3D coordinates (which you would then provide). What is the total length of the path?" This adds a layer of spatial reasoning, measurement, and problem-solving. Conversely, to decrease the difficulty, you could simplify by focusing on a single, easily identifiable attribute of one object: "What color is the large car on the right?" or "How many wheels does the green vehicle have?" This directs attention to immediate visual recognition without requiring complex interpretation.""",
		difficulty_level=3,
		image_context="synthetic scene",
	)

	result = agent.forward(generation_input)
	code_result = result["code_result"]
	metadata = {
		"adapted_question": code_result.adapted_question,
		"recommended_libraries": result["recommended_libraries"],
		"difficulty_params": result["difficulty_params"],
		"image_context": generation_input.image_context,
	}
	print("🧠 Adapted question:", code_result.adapted_question)
	print("🧾 Code explanation:", code_result.code_explanation)

	save_generated_code(code_result.generated_code, metadata)


if __name__ == "__main__":
	main()

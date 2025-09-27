"""
MathVista Question Generation & Verification System
DSPy-based Coding Agent for Mathematical Visual Question Generation

This module implements a comprehensive system for generating programmatic images
with corresponding questions and ground truth answers using the DSPy framework.
"""

import dspy
import os
import json
import re
import traceback
import subprocess
import sys
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings
import dotenv
dotenv.load_dotenv()
warnings.filterwarnings('ignore')


class ImageContext(Enum):
    """Enumeration of supported image contexts"""
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
    """Input specification for the generation system"""
    initial_question: str
    reference_image_path: Optional[str]
    image_description: str
    difficulty_control: str
    difficulty_level: int
    image_context: str


@dataclass
class GenerationOutput:
    """Output specification for the generation system"""
    question: str
    image_path: str
    ground_truth: str
    metadata: Dict[str, Any]
    generated_code: str


class LibrarySelector:
    """Selects appropriate libraries based on image context"""
    
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
        ImageContext.SYNTHETIC_SCENE: ["blender", "pyvista", "trimesh", "open3d", "matplotlib"],
        ImageContext.TABLE: ["pandas", "matplotlib", "PIL", "plotly"],
        ImageContext.VIOLIN_PLOT: ["seaborn", "matplotlib", "plotly", "pandas"]
    }
    
    @classmethod
    def get_libraries(cls, context: str) -> List[str]:
        """Get recommended libraries for given image context"""
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


# DSPy Signatures
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
    """Generate Python code for mathematical visualization with programmatic ground truth calculation. The code must be complete, executable, and save the image to 'output_path'. Required variables: 'ground_truth' (dict) and 'final_image_path' (str). Use only the recommended libraries and scale complexity based on difficulty level."""
    
    question_analysis = dspy.InputField(desc="Analysis of the original question and adaptation requirements")
    mathematical_concepts = dspy.InputField(desc="Key mathematical concepts to implement in the visualization")
    recommended_libraries = dspy.InputField(desc="Specific libraries to use (matplotlib, numpy, seaborn, pandas, etc.)")
    difficulty_parameters = dspy.InputField(desc="Complexity parameters including num_data_points, decimal_places, visual_elements")
    image_context = dspy.InputField(desc="Type of visualization (bar_chart, function_plot, geometry_diagram, scatter_plot, etc.)")
    initial_question = dspy.InputField(desc="Original mathematical question to base the visualization on")
    difficulty_control = dspy.InputField(desc="Instructions for modifying difficulty level")
    
    generated_code = dspy.OutputField(desc="Complete executable Python code that: 1) Imports required libraries, 2) Generates/creates data programmatically, 3) Creates the visualization, 4) Calculates ground truth mathematically (not using AI), 5) Saves image to output_path, 6) Sets final_image_path=output_path, 7) Sets ground_truth as dictionary with calculated values")
    adapted_question = dspy.OutputField(desc="Mathematical question adapted to match the generated visualization and target difficulty level")
    code_explanation = dspy.OutputField(desc="Clear explanation of what the code does, how ground truth is calculated, and what mathematical concepts are demonstrated")


class CodeVerification(dspy.Signature):
    """Verify the generated code for correctness and quality"""
    generated_code = dspy.InputField()
    adapted_question = dspy.InputField()
    expected_answer = dspy.InputField()
    
    verification_result = dspy.OutputField(desc="Code verification result with any issues identified")
    suggestions = dspy.OutputField(desc="Suggestions for improving the code if needed")
    is_valid = dspy.OutputField(desc="Boolean indicating if code passes verification")


class MathematicalAccuracy(dspy.Signature):
    """Verify mathematical accuracy of generated content"""
    adapted_question = dspy.InputField()
    expected_answer = dspy.InputField()
    generated_code = dspy.InputField()
    
    accuracy_assessment = dspy.OutputField(desc="Assessment of mathematical accuracy")
    ground_truth_verification = dspy.OutputField(desc="Verification of ground truth calculation")
    mathematical_validity = dspy.OutputField(desc="Boolean indicating mathematical validity")


# DSPy Modules
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


class VerifierAgent(dspy.Module):
    """Verifier agent module for validating VLM-generated content"""
    
    def __init__(self, enable_thinking: bool = True):
        super().__init__()
        # Use ChainOfThought only if thinking is enabled (not for thinking models)
        if enable_thinking:
            self.code_verifier = dspy.ChainOfThought(CodeVerification)
            self.math_verifier = dspy.ChainOfThought(MathematicalAccuracy)
        else:
            # For thinking models, use direct signatures without ChainOfThought wrapper
            self.code_verifier = dspy.Predict(CodeVerification)
            self.math_verifier = dspy.Predict(MathematicalAccuracy)
    
    def forward(self, generated_code: str, adapted_question: str, code_explanation: str) -> Dict[str, Any]:
        """Forward pass of the verifier agent for VLM-generated code"""
        
        # Step 1: Verify code quality and syntax
        code_verification = self.code_verifier(
            generated_code=generated_code,
            adapted_question=adapted_question,
            expected_answer=code_explanation
        )
        
        # Step 2: Verify mathematical accuracy and ground truth calculation
        math_verification = self.math_verifier(
            adapted_question=adapted_question,
            expected_answer=code_explanation,
            generated_code=generated_code
        )
        
        # Step 3: Basic syntax check
        syntax_valid = self._check_syntax(generated_code)
        
        return {
            "code_verification": code_verification,
            "math_verification": math_verification,
            "syntax_valid": syntax_valid,
            "overall_validity": (
                # Handle boolean conversion for DSPy output fields
                self._to_bool(code_verification.is_valid) and 
                self._to_bool(math_verification.mathematical_validity) and 
                syntax_valid
            )
        }
    
    def _to_bool(self, value) -> bool:
        """Convert DSPy output field to boolean"""
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            return value.lower() in ['true', 'yes', '1', 'valid', 'correct']
        else:
            return bool(value)
    
    def _check_syntax(self, code: str) -> bool:
        """Basic syntax check for generated code"""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False





class CodeExecutor:
    """Executes generated code safely and captures outputs"""
    
    def __init__(self, output_dir: str = "generated_images"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def install_required_packages(self, libraries: List[str]) -> bool:
        """Install required packages if not available"""
        package_mapping = {
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'plotly': 'plotly',
            'PIL': 'Pillow',
            'pandas': 'pandas',
            'numpy': 'numpy',
            'sympy': 'sympy',
            'networkx': 'networkx',
            'cv2': 'opencv-python',
            'folium': 'folium',
            'pyvista': 'pyvista',
            'trimesh': 'trimesh',
            'open3d': 'open3d'
        }
        
        for lib in libraries:
            if lib in package_mapping:
                try:
                    __import__(lib)
                except ImportError:
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", package_mapping[lib]])
                    except subprocess.CalledProcessError:
                        print(f"Warning: Could not install {package_mapping[lib]}")
                        return False
        return True
    
    def execute_code(self, code: str, filename: str) -> Tuple[bool, str, Any]:
        """Execute generated code and return success status, output path, and ground truth"""
        try:
            # Create a safe execution environment with common imports
            exec_globals = {
                '__builtins__': __builtins__,
                'os': os,
                'sys': sys,
                'math': __import__('math'),
                'output_path': os.path.join(self.output_dir, filename)
            }
            
            # Dynamically import common libraries that might be needed
            try:
                exec_globals['np'] = __import__('numpy')
                exec_globals['numpy'] = exec_globals['np']
            except ImportError:
                pass
                
            try:
                import matplotlib.pyplot as plt
                exec_globals['plt'] = plt
                exec_globals['matplotlib'] = __import__('matplotlib')
            except ImportError:
                pass
                
            try:
                exec_globals['sns'] = __import__('seaborn')
                exec_globals['seaborn'] = exec_globals['sns']
            except ImportError:
                pass
                
            try:
                exec_globals['pd'] = __import__('pandas')
                exec_globals['pandas'] = exec_globals['pd']
            except ImportError:
                pass
                
            try:
                from PIL import Image, ImageDraw, ImageFont
                exec_globals['Image'] = Image
                exec_globals['ImageDraw'] = ImageDraw
                exec_globals['ImageFont'] = ImageFont
                exec_globals['PIL'] = __import__('PIL')
            except ImportError:
                pass
            
            exec_locals = {}
            
            # Execute the code
            exec(code, exec_globals, exec_locals)
            
            # Extract results
            image_path = exec_locals.get('final_image_path', exec_globals['output_path'])
            ground_truth = exec_locals.get('ground_truth', {})
            
            # Verify that image was created
            if isinstance(image_path, str) and os.path.exists(image_path):
                return True, image_path, ground_truth
            else:
                return False, "No image file was created or image path not found", None
            
        except Exception as e:
            error_msg = f"Error executing code: {str(e)}\n{traceback.format_exc()}"
            return False, error_msg, None


def setup_model(model_type: str = "ollama", model_name: str = "qwen3:8b") -> bool:
    """Setup DSPy language model configuration
    
    Returns:
        bool: True if DSPy thinking should be enabled, False if disabled (for thinking models)
    """
    api_key = os.getenv('GEMINI_API_KEY', 'GEMINI_API_KEY')
    # Models that have built-in thinking capabilities and should use think=False
    thinking_models = ['qwen', 'qwen2', 'qwen3', 'deepseek', 'o1']
    
    if model_type.lower() == "ollama":
        lm = dspy.LM(f'ollama_chat/{model_name}', api_base='http://localhost:11434', api_key='', max_tokens=32000)
        print(f"✓ Configured Ollama model ({model_name})")
        
        # Check if this is a thinking model
        has_built_in_thinking = any(thinking_model in model_name.lower() for thinking_model in thinking_models)
        if has_built_in_thinking:
            print(f"🧠 Detected thinking model - will disable internal thinking mode and replace with DSPy chain of thought")
        
    elif model_type.lower() == "gemini":
        if not api_key:
            api_key = os.getenv('GEMINI_API_KEY', 'GEMINI_API_KEY')
        lm = dspy.LM(f'gemini/{model_name}', api_key=api_key, max_tokens=32000)
        print(f"✓ Configured Gemini model ({model_name})")
        has_built_in_thinking = False  # Gemini doesn't have built-in thinking mode
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'ollama' or 'gemini'")
    
    dspy.configure(lm=lm)
    return not has_built_in_thinking  # Return False for thinking models (disable DSPy thinking)


class MathVistaSystem:
    """Main system orchestrating the VLM-based generation pipeline"""
    
    def __init__(self, model_type: str = "ollama", model_name: str = "qwen3:8b", seed: Optional[int] = None):
        """Initialize the system with DSPy configuration"""
        
        self.seed = seed
        
        # Set seed for reproducibility where possible
        if seed is not None:
            import random
            import numpy as np
            random.seed(seed)
            if 'numpy' in sys.modules or 'np' in globals():
                np.random.seed(seed)
        
        # Use the modern DSPy setup
        self.enable_thinking = setup_model(model_type, model_name)
        
        # Initialize agents with thinking configuration
        self.coding_agent = CodingAgent(enable_thinking=self.enable_thinking)
        self.verifier_agent = VerifierAgent(enable_thinking=self.enable_thinking)
        self.code_executor = CodeExecutor()
        
        # Initialize optimization if needed
        self.optimizer = None
    
    # Remove the old configuration methods since we use setup_model now
    
    def set_reproducible_mode(self, seed: int):
        """Enable reproducible mode with given seed"""
        import random
        import numpy as np
        
        self.seed = seed
        random.seed(seed)
        
        try:
            np.random.seed(seed)
        except:
            pass
        
        # Note: Modern DSPy doesn't use dspy.settings for runtime configuration
        print(f"🎲 Set seed to {seed} for reproducible generation")
    
    def generate(self, generation_input: GenerationInput, 
                output_filename: Optional[str] = None) -> GenerationOutput:
        """Generate a complete mathematical visualization with question and answer using VLM"""
        
        if output_filename is None:
            output_filename = f"generated_{generation_input.image_context.replace(' ', '_')}_{generation_input.difficulty_level}.png"
        
        try:
            # Step 1: Run coding agent (which now uses VLM for code generation)
            print("🔄 Analyzing question and generating code...")
            coding_result = self.coding_agent(generation_input)
            
            # Step 2: Run verifier agent
            print("✅ Code generated, running verification...")
            verification_result = self.verifier_agent(
                coding_result["code_result"].generated_code,
                coding_result["code_result"].adapted_question,
                coding_result["code_result"].code_explanation
            )
            
            # Step 3: If verification fails, try to fix the code or regenerate
            max_retries = 3
            retry_count = 0
            
            while not verification_result["overall_validity"] and retry_count < max_retries:
                print(f"⚠️ Code verification failed. Retrying... ({retry_count + 1}/{max_retries})")
                
                # Re-generate code with feedback
                coding_result = self.coding_agent(generation_input)
                verification_result = self.verifier_agent(
                    coding_result["code_result"].generated_code,
                    coding_result["code_result"].adapted_question,
                    coding_result["code_result"].code_explanation
                )
                retry_count += 1
            
            # Step 4: Execute code if verification passes
            if verification_result["overall_validity"]:
                print("✅ Verification passed, executing code...")
                # Try to install required libraries
                self.code_executor.install_required_packages(coding_result["recommended_libraries"])
                
                success, result, ground_truth = self.code_executor.execute_code(
                    coding_result["code_result"].generated_code,
                    output_filename
                )
                
                if success:
                    print(f"🎉 Successfully generated visualization: {result}")
                    # Create final output
                    output = GenerationOutput(
                        question=coding_result["code_result"].adapted_question,
                        image_path=result,
                        ground_truth=str(ground_truth) if ground_truth else "Ground truth calculation failed",
                        metadata={
                            "difficulty_level": generation_input.difficulty_level,
                            "image_context": generation_input.image_context,
                            "recommended_libraries": coding_result["recommended_libraries"],
                            "difficulty_params": coding_result["difficulty_params"],
                            "verification_result": verification_result,
                            "code_explanation": coding_result["code_result"].code_explanation,
                            "retry_count": retry_count
                        },
                        generated_code=coding_result["code_result"].generated_code
                    )
                    
                    return output
                else:
                    raise RuntimeError(f"Code execution failed: {result}")
            else:
                raise ValueError("Generated code failed verification after maximum retries")
                
        except Exception as e:
            print(f"❌ Generation error: {str(e)}")
            raise
    
    def optimize_pipeline(self, training_examples: List[Tuple[GenerationInput, GenerationOutput]]):
        """Optimize the DSPy pipeline using training examples"""
        
        # Define metric for optimization
        def quality_metric(gold, pred, trace=None):
            # This would implement a quality metric based on:
            # - Mathematical accuracy of ground truth
            # - Code quality and executability
            # - Question-image alignment
            # - Difficulty consistency
            score = 0.0
            
            # Check if code executes successfully
            if hasattr(pred, 'generated_code'):
                try:
                    compile(pred.generated_code, '<string>', 'exec')
                    score += 0.3  # 30% for syntactic correctness
                except:
                    pass
            
            # Check if ground truth is calculated
            if hasattr(pred, 'ground_truth') and pred.ground_truth != "Ground truth calculation failed":
                score += 0.4  # 40% for ground truth calculation
            
            # Check question adaptation
            if hasattr(pred, 'question') and pred.question:
                score += 0.3  # 30% for question adaptation
            
            return score
        
        # Use DSPy optimizer
        from dspy.teleprompt import BootstrapFewShot
        
        self.optimizer = BootstrapFewShot(metric=quality_metric)
        
        # Optimize the coding agent
        self.coding_agent = self.optimizer.compile(
            self.coding_agent, 
            trainset=training_examples
        )
    
    def batch_generate(self, inputs: List[GenerationInput]) -> List[GenerationOutput]:
        """Generate multiple visualizations in batch"""
        results = []
        
        for i, input_data in enumerate(inputs):
            try:
                output = self.generate(input_data, f"batch_generated_{i}.png")
                results.append(output)
                print(f"Successfully generated {i+1}/{len(inputs)}")
            except Exception as e:
                print(f"Failed to generate for input {i}: {str(e)}")
                continue
        
        return results
    
    def evaluate_generation_quality(self, output: GenerationOutput) -> Dict[str, float]:
        """Evaluate the quality of a generated output"""
        quality_scores = {
            'code_executability': 0.0,
            'ground_truth_validity': 0.0,
            'question_coherence': 0.0,
            'image_generation': 0.0
        }
        
        # Check if code compiles
        try:
            compile(output.generated_code, '<string>', 'exec')
            quality_scores['code_executability'] = 1.0
        except:
            quality_scores['code_executability'] = 0.0
        
        # Check if image exists
        if os.path.exists(output.image_path):
            quality_scores['image_generation'] = 1.0
        
        # Check if ground truth is meaningful
        if output.ground_truth and output.ground_truth != "Ground truth calculation failed":
            quality_scores['ground_truth_validity'] = 1.0
        
        # Check if question is non-empty and coherent
        if output.question and len(output.question.strip()) > 10:
            quality_scores['question_coherence'] = 1.0
        
        # Overall score
        overall_score = sum(quality_scores.values()) / len(quality_scores)
        quality_scores['overall'] = overall_score
        
        return quality_scores


# Export main classes and functions
__all__ = [
    'MathVistaSystem',
    'CodingAgent', 
    'VerifierAgent',
    'GenerationInput',
    'GenerationOutput',
    'ImageContext',
    'LibrarySelector',
    'DifficultyController',
    'CodeExecutor'
]


if __name__ == "__main__":
    # Example usage with VLM-based code generation using modern DSPy setup
    system = MathVistaSystem(model_type="gemini", model_name="gemini-2.5-flash", seed=42)
    
    # Example input
    input_data = GenerationInput(
        initial_question="Is the number of tiny gray bicycles that are on the left side of the brown metal sedan greater than the number of things that are to the left of the tiny green bicycle?",
        reference_image_path="img/image_1.png",
        image_description="""The image is a 3D rendering of several vehicles on a plain, light-colored surface. The lighting is from the upper left, casting subtle shadows. The vehicles are: 
        A metallic red motorcycle: Highly stylized with a streamlined, almost futuristic design. It has intricate, chrome-like details, and a light-blue or cyan seat and wheel rims. It is positioned near the center-left of the image.
        A metallic gold sedan: A large, four-door car with a sleek, reflective gold finish. It's positioned on the right side of the image, facing slightly left.
        A small, pink sedan: A small, simple-looking car with a pink body and a light-blue roof. It's located behind the red motorcycle and the green bicycle.
        A green bicycle: A classic-style bicycle with a bright green frame and a thin, turquoise saddle and wheels. It is situated to the left of the gray dirt bike.
        A gray dirt bike: A large, uncolored or matte gray dirt bike with a green front fender and blue wheels. It's the largest vehicle in the group and is positioned in the center of the image.
        Two joined bicycles: A unique, low-profile vehicle consisting of two bicycle-like frames joined together side-by-side, lying on the ground. One side is blue and the other is a metallic orange or bronze color. This vehicle is in the foreground, at the bottom of the image.
        The overall scene has a somewhat surreal or artistic feel due to the mix of vehicle styles and the varied, often reflective, materials.""",
        difficulty_control="""To adjust the difficulty for a question about this image, consider manipulating the context and relationships between the objects. For instance, to increase the difficulty, you could introduce a narrative or a physical interaction: "Imagine these vehicles are part of a race, and the two joined bicycles are obstacles. The red motorcycle needs to choose the shortest path around them to reach the finish line, which is marked by the golden car. Calculate the minimal distance if each unit represents one meter, and the vehicles are currently at specific 3D coordinates (which you would then provide). What is the total length of the path?" This adds a layer of spatial reasoning, measurement, and problem-solving. Conversely, to decrease the difficulty, you could simplify by focusing on a single, easily identifiable attribute of one object: "What color is the large car on the right?" or "How many wheels does the green vehicle have?" This directs attention to immediate visual recognition without requiring complex interpretation.""",
        difficulty_level=3,
        image_context="synthetic scene"
    )
    
    # Generate output using VLM
    try:
        output = system.generate(input_data)
        print(f"Generated question: {output.question}")
        print(f"Image saved to: {output.image_path}")
        print(f"Ground truth: {output.ground_truth}")
        print(f"Code explanation: {output.metadata.get('code_explanation', 'N/A')}")
        
        # Evaluate quality
        quality_scores = system.evaluate_generation_quality(output)
        print(f"Quality scores: {quality_scores}")
        
    except Exception as e:
        print(f"Generation failed: {str(e)}")
        
    # # Example batch generation
    # batch_inputs = [
    #     GenerationInput(
    #         initial_question="What is the correlation between X and Y?",
    #         reference_image_path=None,
    #         image_description="Scatter plot showing positive correlation",
    #         difficulty_control="Add multiple regression lines and statistical measures",
    #         difficulty_level=4,
    #         image_context="scatter_plot"
    #     ),
    #     GenerationInput(
    #         initial_question="Which category has the highest value?",
    #         reference_image_path=None,
    #         image_description="Bar chart with 5 categories",
    #         difficulty_control="Add error bars and percentage labels",
    #         difficulty_level=2,
    #         image_context="bar_chart"
    #     )
    # ]
    
    # try:
    #     batch_results = system.batch_generate(batch_inputs)
    #     print(f"Successfully generated {len(batch_results)} visualizations")
        
    #     for i, result in enumerate(batch_results):
    #         quality = system.evaluate_generation_quality(result)
    #         print(f"Batch item {i+1} quality score: {quality['overall']:.2f}")
            
    except Exception as e:
        print(f"Batch generation failed: {str(e)}")

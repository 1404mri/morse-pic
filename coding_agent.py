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
    ABSTRACT_SCENE = "abstract_scene"
    BAR_CHART = "bar_chart"
    DOCUMENT_IMAGE = "document_image"
    FUNCTION_PLOT = "function_plot"
    GEOMETRY_DIAGRAM = "geometry_diagram"
    LINE_PLOT = "line_plot"
    MAP_CHART = "map_chart"
    PIE_CHART = "pie_chart"
    PUZZLE_TEST = "puzzle_test"
    SCATTER_PLOT = "scatter_plot"
    SCIENTIFIC_FIGURE = "scientific_figure"
    SYNTHETIC_SCENE = "synthetic_scene"
    TABLE = "table"
    VIOLIN_PLOT = "violin_plot"


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
    
    def __init__(self):
        super().__init__()
        self.question_analyzer = dspy.ChainOfThought(QuestionAnalysis)
        self.code_generator = dspy.ChainOfThought(CodeGeneration)
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
    
    def __init__(self):
        super().__init__()
        self.code_verifier = dspy.ChainOfThought(CodeVerification)
        self.math_verifier = dspy.ChainOfThought(MathematicalAccuracy)
    
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
            "overall_validity": (code_verification.is_valid and 
                               math_verification.mathematical_validity and 
                               syntax_valid)
        }
    
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


class MathVistaSystem:
    """Main system orchestrating the VLM-based generation pipeline"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", seed: Optional[int] = None):
        """Initialize the system with DSPy configuration"""
        
        self.seed = seed
        
        # Set seed for reproducibility where possible
        if seed is not None:
            import random
            import numpy as np
            random.seed(seed)
            if 'numpy' in sys.modules or 'np' in globals():
                np.random.seed(seed)
        
        # Configure DSPy with the specified model and seed
        # This would be configured based on whether using Ollama or Gemini
        lm_config = {"model": model_name}
        if seed is not None:
            # Try to set seed/temperature for reproducibility
            lm_config.update({
                "temperature": 0.0,  # Lower temperature for more deterministic responses
                "seed": seed  # Some models support seed parameter
            })
        
        try:
            dspy.settings.configure(lm=dspy.OpenAI(**lm_config))
        except TypeError:
            # Fallback if seed parameter not supported
            dspy.settings.configure(lm=dspy.OpenAI(model=model_name, temperature=0.0))
        
        # Initialize agents
        self.coding_agent = CodingAgent()
        self.verifier_agent = VerifierAgent()
        self.code_executor = CodeExecutor()
        
        # Initialize optimization if needed
        self.optimizer = None
    
    def configure_for_ollama(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        """Configure DSPy to use Ollama models"""
        try:
            import ollama
            # Configure DSPy for Ollama with reproducibility settings
            lm_config = {
                "model": model, 
                "base_url": base_url,
                "temperature": 0.0  # Set temperature to 0 for reproducibility
            }
            if self.seed is not None:
                lm_config["seed"] = self.seed
            
            dspy.settings.configure(lm=dspy.OllamaLocal(**lm_config))
        except ImportError:
            print("Ollama not available. Install with: pip install ollama")
        except TypeError:
            # Fallback if some parameters not supported
            dspy.settings.configure(lm=dspy.OllamaLocal(model=model, base_url=base_url))
    
    def configure_for_gemini(self, api_key: str, model: str = "gemini-pro"):
        """Configure DSPy to use Gemini models"""
        try:
            # Configure DSPy for Gemini with reproducibility settings
            lm_config = {
                "model": model, 
                "api_key": api_key,
                "temperature": 0.0  # Set temperature to 0 for reproducibility
            }
            if self.seed is not None:
                # Note: Gemini may not support seed parameter directly
                # but temperature=0 helps with reproducibility
                pass
            
            dspy.settings.configure(lm=dspy.Gemini(**lm_config))
        except Exception as e:
            print(f"Gemini configuration failed: {e}")
    
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
        
        # Update DSPy configuration for reproducibility
        current_lm = dspy.settings.lm
        if hasattr(current_lm, 'kwargs'):
            current_lm.kwargs.update({
                'temperature': 0.0,
                'seed': seed
            })
    
    def generate(self, generation_input: GenerationInput, 
                output_filename: Optional[str] = None) -> GenerationOutput:
        """Generate a complete mathematical visualization with question and answer using VLM"""
        
        if output_filename is None:
            output_filename = f"generated_{generation_input.image_context}_{generation_input.difficulty_level}.png"
        
        # Step 1: Run coding agent (which now uses VLM for code generation)
        coding_result = self.coding_agent(generation_input)
        
        # Step 2: Run verifier agent
        verification_result = self.verifier_agent(
            coding_result["code_result"].generated_code,
            coding_result["code_result"].adapted_question,
            coding_result["code_result"].code_explanation
        )
        
        # Step 3: If verification fails, try to fix the code or regenerate
        max_retries = 3
        retry_count = 0
        
        while not verification_result["overall_validity"] and retry_count < max_retries:
            print(f"Code verification failed. Retrying... ({retry_count + 1}/{max_retries})")
            
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
            # Try to install required libraries
            self.code_executor.install_required_packages(coding_result["recommended_libraries"])
            
            success, result, ground_truth = self.code_executor.execute_code(
                coding_result["code_result"].generated_code,
                output_filename
            )
            
            if success:
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
    # Example usage with VLM-based code generation
    system = MathVistaSystem(seed=42)  # Set seed for reproducibility
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set the GOOGLE_API_KEY environment variable for Gemini access.")
    # Configure for different models
    # system.configure_for_ollama(model="codellama")  # For Ollama
    system.configure_for_gemini(api_key=api_key)  # For Gemini

    # Example input
    input_data = GenerationInput(
        initial_question="What is the area of the triangle?",
        reference_image_path=None,
        image_description="A right triangle with sides 3, 4, and 5",
        difficulty_control="Make the calculation more complex by adding angle measurements and multiple triangles",
        difficulty_level=3,
        image_context="geometry_diagram"
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
        
    # Example batch generation
    batch_inputs = [
        GenerationInput(
            initial_question="What is the correlation between X and Y?",
            reference_image_path=None,
            image_description="Scatter plot showing positive correlation",
            difficulty_control="Add multiple regression lines and statistical measures",
            difficulty_level=4,
            image_context="scatter_plot"
        ),
        GenerationInput(
            initial_question="Which category has the highest value?",
            reference_image_path=None,
            image_description="Bar chart with 5 categories",
            difficulty_control="Add error bars and percentage labels",
            difficulty_level=2,
            image_context="bar_chart"
        )
    ]
    
    try:
        batch_results = system.batch_generate(batch_inputs)
        print(f"Successfully generated {len(batch_results)} visualizations")
        
        for i, result in enumerate(batch_results):
            quality = system.evaluate_generation_quality(result)
            print(f"Batch item {i+1} quality score: {quality['overall']:.2f}")
            
    except Exception as e:
        print(f"Batch generation failed: {str(e)}")

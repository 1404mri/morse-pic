# MORSE-PIC ✨

MORSE-PIC (Mathematical Oriented Reasoning & Synthetic Environment – Programmable Image Creator) is a DSPy-based coding agent that turns textual math questions into runnable Python programs which draw visualizations, compute ground-truth answers, and store metadata. You can use it interactively through the `MathVistaSystem` pipeline or run a standalone generator that writes the raw Python code to disk.

---

## 🚀 Key Capabilities

- **LLM-driven code generation:** Uses DSPy signatures (`QuestionAnalysis`, `CodeGeneration`) to prompt an LLM for executable matplotlib/seaborn/plotly code.
- **Difficulty-aware outputs:** Automatically scales data volume, precision, and visual complexity based on the requested difficulty level.
- **Multiple image contexts:** Supports charts, tables, geometry diagrams, synthetic scenes, and more via a context-aware library selector.
- **Ground-truth aware:** Generated programs must compute answers numerically (no language-model guessing) and surface them as dictionaries.
- **Reproducible pipeline:** Optional seeding keeps DSPy chains deterministic when the backend LLM supports it.
- **Verifier & executor loop:** (In `coding_agent.py`) validates syntax/mathematical correctness, retries on failure, and executes the generated code to produce image files.

---

## 🏗️ Project Layout

```
MORSE-PIC/
├── coding_agent.py          # Full MathVistaSystem pipeline (generation + verification + execution)
├── generate_image.py        # Standalone CodingAgent copy; saves raw LLM-generated code to ./generated_code
├── test_coding_agent.py     # Smoke test for CodingAgent.forward
├── img/                     # Sample input images
├── generated_code/          # Auto-created; stores generated scripts & metadata
├── requirements.txt         # Python dependencies
└── README.md                # This guide
```

---

## 🧩 Prerequisites

- **Python** 3.10+ (tested with 3.12)
- **pip / virtualenv** for dependency management
- **LLM backend** (choose one):
	- Google Gemini (`gemini-1.5-flash`, `gemini-2.5-flash`, etc.)
	- Ollama local models (`qwen3`, `deepseek`, `o1` style models)
- **API keys / services**
	- `GEMINI_API_KEY` for Gemini models
	- Ollama must be running locally (`ollama serve`) if you select that backend

---

## 🔧 Installation

```bash
git clone <repository-url>
cd MORSE-PIC
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file (or export variables in your shell):

```env
GEMINI_API_KEY=your-google-gemini-key
MODEL_TYPE=gemini              # or "ollama"
MODEL_NAME=gemini-1.5-flash    # or an Ollama model name, e.g. "qwen3:8b"
```

> **Tip:** When using Ollama, make sure to pull the model first, e.g. `ollama pull qwen3:8b`.

---

## 🧪 Quick Smoke Test

Run the lightweight unit test to confirm the `CodingAgent` can generate a response:

```bash
python test_coding_agent.py
```

This loads the configured LLM, calls `CodingAgent.forward`, and verifies the structure of the outputs.

---

## 🧠 MathVistaSystem Workflow (`coding_agent.py`)

`MathVistaSystem` orchestrates a full end-to-end generation:

1. **CodingAgent** analyzes the input question, selects libraries, and asks the LLM for Python code.
2. **VerifierAgent** double-checks syntax and mathematical validity (with DSPy Chain-of-Thought).
3. **CodeExecutor** installs missing packages, executes the generated code, and collects artifacts
4. **Outputs** include the adapted question, image path, ground-truth dictionary, metadata, and quality scores.

### Running the pipeline

```bash
python coding_agent.py
```

What happens:

- Loads the default Gemini model (`gemini-2.5-flash`) unless overridden via env vars
- Generates an adapted question & explanation for the sample synthetic scene prompt
- Saves the rendered plot to `generated_images/`
- Prints quality metrics and ground-truth values

### Customizing Inputs

Replace the `GenerationInput` block near the bottom of `coding_agent.py` with your own:

```python
input_data = GenerationInput(
		initial_question="What is the correlation between study time and score?",
		reference_image_path=None,
		image_description="Scatter plot with moderate positive correlation",
		difficulty_control="Add regression lines and annotate key points",
		difficulty_level=4,
		image_context="scatter plot",
)
```

### Choosing an LLM backend

`MathVistaSystem` automatically calls `setup_model(model_type, model_name)` during initialization. Configure via env vars:

```bash
export MODEL_TYPE=ollama
export MODEL_NAME=qwen3:8b
```

Gemini requires `GEMINI_API_KEY`. Ollama ignores the API key but expects the server to be running locally.

### Reproducible Runs

You can provide a `seed` to the constructor to align random libraries and numpy calls. Note that absolute determinism depends on the LLM backend:

```python
system = MathVistaSystem(model_type="gemini", model_name="gemini-2.5-flash", seed=42)
```

---

## 🗂️ Standalone Code Exporter (`generate_image.py`)

Sometimes you just want the raw LLM-generated script without running verifiers or executing it. `generate_image.py`:

- Re-implements the `CodingAgent` logic inline (no imports from `coding_agent.py`)
- Calls the agent once with a synthetic-scene prompt
- Extracts the Python code block (ignoring markdown formatting) and writes it to `generated_code/generated_<context>.py`
- Stores metadata alongside the script in `generated_code/metadata.json`

### Usage

```bash
python generate_image.py
```

Outputs:

- `generated_code/generated_synthetic_scene.py` – cleaned Python script ready to run
- `generated_code/metadata.json` – adapted question, libraries, difficulty parameters, and context

Customize the prompt or difficulty by editing the `generation_input` definition near the bottom of the file.

---

## 🧪 Testing & Validation

| Command | Description |
|---------|-------------|
| `python test_coding_agent.py` | Basic contract test for `CodingAgent.forward` |
| `python coding_agent.py` | Full generation + verification + execution pipeline |
| `python generate_image.py` | Export raw generated code without running it |

> **Note:** The tests rely on a configured LLM. If you only need structure validation without calling the API, consider mocking DSPy in future enhancements.

---

## ⚙️ Configuration Reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `GEMINI_API_KEY` | _required for Gemini_ | Auth token for Google Gemini models |
| `MODEL_TYPE` | `gemini` | Selects LLM provider (`gemini` or `ollama`) |
| `MODEL_NAME` | `gemini-1.5-flash` | Specific model to load |
| `OLLAMA_HOST` | `http://localhost:11434` | Override Ollama base URL (set manually if needed) |

Project scripts also install missing Python libs on the fly (e.g., `opencv-python`, `folium`). Ensure your environment permits pip installs.

---

## 🛣️ Roadmap / Ideas

- Cache verified prompts/examples to bootstrap DSPy optimizers faster
- Expand `LibrarySelector` contexts (heatmaps, 3D plots, maps)
- Package as a CLI with subcommands (`generate`, `verify`, `batch`)
- Add integration tests that verify generated images & ground truth against deterministic fixtures
- Support Azure OpenAI / Anthropic backends via additional `setup_model` branches

Contributions are welcome—open issues or submit pull requests with detailed descriptions.

---

## 📄 License

_Specify your license here (e.g., MIT, Apache 2.0)._

If no license is provided, assume the code is proprietary to the repository owner.

---

## 🙌 Acknowledgements

- Built on top of [DSPy](https://github.com/stanfordnlp/dspy) for structured prompt synthesis
- Uses Python’s scientific stack (matplotlib, seaborn, pandas, numpy, sympy, etc.) for visualization & computation
- Inspired by large-scale visual reasoning benchmarks such as MathVista

Enjoy generating math-rich visuals programmatically! 🎨📐

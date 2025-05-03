# Research Paper Evaluation System

A comprehensive system for evaluating research papers for publishability and recommending suitable conferences.

## Overview

This system evaluates whether a research paper is publishable and, if so, suggests the most suitable conference among: NeurIPS, ICML, ICLR, AAAI, or ACL.

The architecture consists of three main components:

1. **Graph of Thought (GoT) Filter**: Extracts core claims and checks for redundancy/plagiarism
2. **Retrieval-Augmented Generation (RAG)**: Simulates retrieval of similar papers
3. **Tree of Thought (ToT) Reasoning**: Performs multi-path reasoning to evaluate the paper

## Features

- **Originality Assessment**: Evaluates the originality of claims in the paper
- **Plagiarism Detection**: Identifies potentially redundant or plagiarized content
- **Multi-path Reasoning**: Uses Tree of Thought approach to explore different evaluation perspectives
- **Conference Recommendation**: Suggests the most suitable conference for publication
- **Final Decision**: Provides a clear Accept/Reject/Needs Revision decision with reasoning
- **PDF Processing**: Extracts text from PDF files for evaluation
- **Evaluation Metrics**: Calculates accuracy, precision, recall, and F1 score
- **Visualization**: Generates confusion matrix for model performance analysis

## Architecture

### A. Graph of Thought (GoT) Filter
- Extracts the core claims of the input paper using an LLM
- Compares these claims with similar papers retrieved using a RAG step
- Checks semantic similarity between claims and publication years
- Flags redundant or plagiarized claims
- Terminates early with rejection if a majority of claims are redundant

### B. Retrieval-Augmented Generation (RAG)
- Simulates retrieval using a hardcoded list of previous papers
- Each paper includes title, abstract, claims, and published year

### C. Tree of Thought (ToT) Reasoning
- Generates 3 reasoning paths per thought node
- Explores paths through 3 layers of reasoning
- Uses a Critic LLM to score/prune paths based on clarity, novelty, and relevance
- Retains the top 2 branches at each level

### D. Final Evaluation
- Calculates Originality Score (0-100)
- Sets Plagiarism Flag (Yes/No)
- Identifies Best Reasoning Path from ToT
- Makes Final Decision: Accept / Reject / Needs Revision
- Recommends a Conference

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`
- Optional: Hugging Face API token for using their inference API

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/paper-evaluator.git
   cd paper-evaluator
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables (optional):
   Create a `.env` file with:
   ```
   HUGGINGFACE_API_TOKEN=your_huggingface_token_here
   ```

4. Create directories for train and test papers:
   ```
   mkdir train_papers
   mkdir test_papers
   ```

## Usage

### Basic Usage with Text Input

```python
from paper_evaluator import PaperEvaluator

# Create an evaluator with a specific model type
evaluator = PaperEvaluator(model_type="simulated")  # Options: "simulated", "huggingface", "local"

# Evaluate a paper
paper_content = """
Title: Your Paper Title

Abstract:
Your paper abstract...

Introduction:
Your introduction...
"""

# Get evaluation results
results = evaluator.evaluate_paper(paper_content)

# Print results
print(f"Originality Score: {results['originality_score']:.1f}/100")
print(f"Plagiarism Flag: {results['plagiarism_flag']}")
print(f"Decision: {results['decision']}")
print(f"Recommended Conference: {results['recommended_conference']}")
```

### Processing PDF Files

```python
from pdf_processor import PDFProcessor

# Create a processor
processor = PDFProcessor(
    train_dir="train_papers",
    test_dir="test_papers",
    model_type="simulated"
)

# Run evaluation with ground truth
results = processor.run_evaluation(ground_truth_file="sample_ground_truth.json")

# Print metrics
if results["metrics"]:
    print("\n===== EVALUATION METRICS =====")
    for metric, value in results["metrics"].items():
        print(f"{metric.capitalize()}: {value:.4f}")
```

### Running the Demo

```
# Run with simulated responses (default)
python paper_evaluator.py

# Run with Hugging Face API (requires API token)
python paper_evaluator.py huggingface

# Run with local model (requires more resources)
python paper_evaluator.py local

# Process PDF files and evaluate
python pdf_processor.py --train_dir train_papers --test_dir test_papers --model_type simulated --ground_truth sample_ground_truth.json
```

## Model Options

The system supports three model types:

1. **Simulated** (default): Uses simulated responses for demonstration purposes
2. **Hugging Face**: Uses the Hugging Face Inference API with the Mistral-7B-Instruct model
3. **Local**: Loads and runs the Mistral-7B-Instruct model locally (requires more resources)

## Evaluation Metrics

The system calculates the following metrics when ground truth data is available:

- **Accuracy**: Proportion of correct predictions
- **Precision**: Proportion of positive identifications that were actually correct
- **Recall**: Proportion of actual positives that were identified correctly
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of true positives, false positives, true negatives, and false negatives

## Ground Truth Format

The ground truth file should be a JSON file with the following structure:

```json
{
  "test_papers/paper1.pdf": {
    "decision": "Accept",
    "conference": "NeurIPS",
    "originality_score": 95,
    "plagiarism_flag": "No"
  },
  "test_papers/paper2.pdf": {
    "decision": "Reject",
    "conference": "N/A",
    "originality_score": 30,
    "plagiarism_flag": "Yes"
  }
}
```

## Customization

- **Sample Papers**: Modify the `SAMPLE_PAPERS` list in `paper_evaluator.py` to include your own reference papers
- **Conferences**: Change the `conferences` list in the `PaperEvaluator` class to include different conferences
- **Model Selection**: Choose between simulated, Hugging Face, or local model based on your needs

## License

MIT

## Acknowledgements

- This system is inspired by the Graph of Thought and Tree of Thought reasoning frameworks
- Sample papers are from notable research in machine learning and NLP
- Uses Mistral-7B-Instruct as the free LLM alternative 
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
from pypdf import PdfReader
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from paper_evaluator import PaperEvaluator

class PDFProcessor:
    """
    Process PDF files and evaluate the paper evaluator model.
    """
    
    def __init__(self, papers_dir: str, model_type: str = "simulated"):
        self.papers_dir = Path(papers_dir)
        self.model_type = model_type

        if not self.papers_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.papers_dir}")

        print(f"Loading training papers from: {self.papers_dir}")
        self.training_papers = self.process_pdf_files(self.papers_dir)

        try:
            self.evaluator = PaperEvaluator(
                model_type=model_type,
                training_papers=self.training_papers  # ← key addition
            )
            print(f"Initialized PaperEvaluator with model type: {model_type}")
        except Exception as e:
            print(f"ERROR initializing PaperEvaluator: {str(e)}")
            raise

        
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            reader = PdfReader(str(pdf_path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def process_pdf_files(self, directory):
        papers = []
        pdf_dir = Path(directory)

        for pdf_path in pdf_dir.glob("*.pdf"):
            try:
                extracted_text = self.extract_text_from_pdf(pdf_path)
                papers.append({
                    "text": extracted_text,
                    "file_path": str(pdf_path),
                    "title": pdf_path.stem,  # Using filename as placeholder
                    "abstract": "",           # TODO: Implement abstract extraction
                    "claims": []              # TODO: Implement claim extraction
                })
            except Exception as e:
                print(f"Failed to process {pdf_path.name}: {e}")

        return papers

    
    def evaluate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate a list of papers using the paper evaluator.
        
        Args:
            papers: List of papers with file paths and text
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for paper in papers:
            print(f"Evaluating {paper['file_path']}...")
            evaluation = self.evaluator.evaluate_paper(paper["text"])
            
            results.append({
                "file_path": paper["file_path"],
                "evaluation": evaluation
            })
        
        return results
    
    def load_ground_truth(self, ground_truth_file: str) -> Dict[str, Dict[str, Any]]:
        """
        Load ground truth data from a JSON file.
        
        Args:
            ground_truth_file: Path to the ground truth JSON file
            
        Returns:
            Dictionary mapping file paths to ground truth data
        """
        try:
            with open(ground_truth_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Ground truth file {ground_truth_file} not found.")
            return {}
    
    def calculate_metrics(self, results: List[Dict[str, Any]], ground_truth: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate evaluation metrics by comparing results with ground truth.
        
        Args:
            results: List of evaluation results
            ground_truth: Dictionary mapping file paths to ground truth data
            
        Returns:
            Dictionary of metrics
        """
        if not ground_truth:
            print("No ground truth data available. Skipping metrics calculation.")
            return {}
        
        # Prepare data for metrics calculation
        y_true = []
        y_pred = []
        
        for result in results:
            file_path = result["file_path"]
            if file_path in ground_truth:
                # Convert decision to binary (Accept=1, Reject/Needs Revision=0)
                true_decision = 1 if ground_truth[file_path].get("decision", "").lower() == "accept" else 0
                pred_decision = 1 if result["evaluation"]["decision"].lower() == "accept" else 0
                
                y_true.append(true_decision)
                y_pred.append(pred_decision)
        
        if not y_true or not y_pred:
            print("No matching ground truth data found. Skipping metrics calculation.")
            return {}
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Reject/Revise', 'Accept'],
                    yticklabels=['Reject/Revise', 'Accept'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        return metrics
    
    def run_evaluation(self, ground_truth_file: str = None) -> Dict[str, Any]:
        """
        Run the complete evaluation process.
        
        Args:
            ground_truth_file: Path to the ground truth JSON file (optional)
            
        Returns:
            Dictionary with evaluation results and metrics
        """
        # Process all papers
        print("Processing papers...")
        papers = self.process_pdf_files(self.papers_dir)
        
        if not papers:
            print("No papers were successfully processed. Stopping evaluation.")
            return {"test_results": [], "metrics": {}}
        
        # Evaluate papers
        print("Evaluating papers...")
        evaluation_results = self.evaluate_papers(papers)
        
        # Load ground truth if available
        ground_truth = {}
        if ground_truth_file:
            ground_truth = self.load_ground_truth(ground_truth_file)
        
        # Calculate metrics
        metrics = self.calculate_metrics(evaluation_results, ground_truth)
        
        # Save results
        output = {
            "test_results": evaluation_results,
            "metrics": metrics
        }
        
        # Debug output to verify the data before saving
        print(f"Saving {len(evaluation_results)} evaluation results and {len(metrics)} metrics to evaluation_results.json")
        
        try:
            output_file = "evaluation_results.json"
            with open(output_file, "w") as f:
                json.dump(output, f, indent=2)
            print(f"Successfully saved results to {os.path.abspath(output_file)}")
        except Exception as e:
            print(f"Error saving results to evaluation_results.json: {e}")
        
        return output

def main():
    """Main function to demonstrate the PDF processor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process and evaluate research papers in PDF format")
    parser.add_argument("--papers_dir", type=str, required=True, 
                        help="Directory containing PDF papers for both training and testing")
    parser.add_argument("--model_type", type=str, default="simulated", 
                        choices=["simulated", "huggingface", "local"], 
                        help="Type of model to use for evaluation")
    parser.add_argument("--ground_truth", type=str, default=None, 
                        help="Path to ground truth JSON file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = PDFProcessor(args.papers_dir, args.model_type)
    
    # Run evaluation
    results = processor.run_evaluation(args.ground_truth)
    
    # Print metrics
    if results["metrics"]:
        print("\n===== EVALUATION METRICS =====")
        for metric, value in results["metrics"].items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        print("\nConfusion matrix saved as 'confusion_matrix.png'")
    
    print("\nEvaluation results saved to 'evaluation_results.json'")

if __name__ == "__main__":
    main()
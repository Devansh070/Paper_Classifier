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
    
    def __init__(self, train_dir: str, test_dir: str, model_type: str = "simulated"):
        """
        Initialize the PDF processor.
        
        Args:
            train_dir: Directory containing training PDF files
            test_dir: Directory containing test PDF files
            model_type: Type of model to use for evaluation
        """
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)
        self.model_type = model_type
        self.evaluator = PaperEvaluator(model_type=model_type)
        
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
    
    def process_pdf_files(self, directory: Path) -> List[Dict[str, Any]]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory: Directory containing PDF files
            
        Returns:
            List of dictionaries with file paths and extracted text
        """
        pdf_files = list(directory.glob("*.pdf"))
        results = []
        
        for pdf_path in pdf_files:
            text = self.extract_text_from_pdf(pdf_path)
            if text:
                results.append({
                    "file_path": str(pdf_path),
                    "text": text
                })
        
        return results
    
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
        # Process training papers
        print("Processing training papers...")
        train_papers = self.process_pdf_files(self.train_dir)
        
        # Process test papers
        print("Processing test papers...")
        test_papers = self.process_pdf_files(self.test_dir)
        
        # Evaluate test papers
        print("Evaluating test papers...")
        test_results = self.evaluate_papers(test_papers)
        
        # Load ground truth if available
        ground_truth = {}
        if ground_truth_file:
            ground_truth = self.load_ground_truth(ground_truth_file)
        
        # Calculate metrics
        metrics = self.calculate_metrics(test_results, ground_truth)
        
        # Save results
        output = {
            "test_results": test_results,
            "metrics": metrics
        }
        
        with open("evaluation_results.json", "w") as f:
            json.dump(output, f, indent=2)
        
        return output

def main():
    """Main function to demonstrate the PDF processor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process and evaluate research papers in PDF format")
    parser.add_argument("--train_dir", type=str, default="train_papers", help="Directory containing training PDF files")
    parser.add_argument("--test_dir", type=str, default="test_papers", help="Directory containing test PDF files")
    parser.add_argument("--model_type", type=str, default="simulated", choices=["simulated", "huggingface", "local"], 
                        help="Type of model to use for evaluation")
    parser.add_argument("--ground_truth", type=str, default=None, help="Path to ground truth JSON file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = PDFProcessor(args.train_dir, args.test_dir, args.model_type)
    
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
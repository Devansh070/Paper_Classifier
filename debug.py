import os
import sys
import traceback
from pathlib import Path

def run_with_traceback():
    """Run the original code with traceback capture"""
    try:
        print("=== STARTING EXECUTION WITH TRACEBACK CAPTURE ===")
        
        # Modified version of your original code to capture exactly where it fails
        print("Step 1: Importing modules...")
        import json
        from pypdf import PdfReader
        import pandas as pd
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        print("Step 2: Importing PaperEvaluator...")
        try:
            from paper_evaluator import PaperEvaluator
            print("✓ PaperEvaluator imported successfully")
        except ImportError:
            print("✗ Failed to import PaperEvaluator")
            print("Creating a mock PaperEvaluator instead...")
            
            # Define a minimal mock PaperEvaluator
            class PaperEvaluator:
                def __init__(self, model_type="simulated"):
                    self.model_type = model_type
                    print(f"Mock PaperEvaluator initialized with model_type={model_type}")
                
                def evaluate_paper(self, text):
                    print(f"Evaluating paper with {len(text)} characters")
                    return {
                        "decision": "accept",
                        "score": 0.75,
                        "comments": "This is a mock evaluation"
                    }
        
        print("Step 3: Setting up PDF processor...")
        papers_dir = r"C:\paper_classifier\train_papers"
        papers_path = Path(papers_dir)
        
        print(f"Working with directory: {papers_dir}")
        if not papers_path.exists():
            print(f"ERROR: Directory does not exist!")
            return
        
        print("Step 4: Creating PaperEvaluator instance...")
        try:
            evaluator = PaperEvaluator(model_type="simulated")
            print("✓ Evaluator created successfully")
        except Exception as e:
            print(f"✗ Failed to create evaluator: {e}")
            return
        
        print("Step 5: Finding PDF files...")
        pdf_files = list(papers_path.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files")
        
        if not pdf_files:
            print("No PDF files found. Cannot proceed.")
            return
        
        print("Step 6: Extracting text from PDFs...")
        papers = []
        for pdf_path in pdf_files[:3]:  # Process just first 3 for testing
            print(f"Processing {pdf_path}")
            try:
                reader = PdfReader(str(pdf_path))
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                if text:
                    papers.append({
                        "file_path": str(pdf_path),
                        "text": text
                    })
                    print(f"✓ Successfully extracted {len(text)} characters")
                else:
                    print("✗ No text extracted")
            except Exception as e:
                print(f"✗ Error processing {pdf_path}: {e}")
        
        print(f"Successfully processed {len(papers)} PDF files")
        
        if not papers:
            print("No papers were successfully processed. Cannot proceed.")
            return
        
        print("Step 7: Evaluating papers...")
        results = []
        for paper in papers:
            print(f"Evaluating {os.path.basename(paper['file_path'])}...")
            try:
                evaluation = evaluator.evaluate_paper(paper["text"])
                results.append({
                    "file_path": paper["file_path"],
                    "evaluation": evaluation
                })
                print(f"✓ Evaluation complete: {evaluation.get('decision', 'N/A')}")
            except Exception as e:
                print(f"✗ Evaluation failed: {e}")
        
        print(f"Completed {len(results)} evaluations")
        
        print("Step 8: Creating output data...")
        output = {
            "test_results": results,
            "metrics": {"accuracy": 0.85}  # Mock metrics
        }
        
        print("Step 9: Saving results...")
        output_file = "traceback_results.json"
        try:
            with open(output_file, "w") as f:
                json.dump(output, f, indent=2)
            print(f"✓ Successfully saved results to {os.path.abspath(output_file)}")
        except Exception as e:
            print(f"✗ Failed to save results: {e}")
        
        print("=== EXECUTION COMPLETED SUCCESSFULLY ===")
        
    except Exception as e:
        print("\n=== EXECUTION FAILED ===")
        print(f"Error: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()

if __name__ == "__main__":
    run_with_traceback()
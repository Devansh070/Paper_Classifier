from paper_evaluator import PaperEvaluator

def test_paper_evaluator():
    """
    Test the PaperEvaluator with a sample paper using the simulated model type.
    This allows testing without requiring a dataset.
    """
    print("Initializing PaperEvaluator with simulated model...")
    evaluator = PaperEvaluator(model_type="simulated", max_workers=2)
    
    # Sample training papers
    training_papers = [
        {
            "title": "Deep Learning for NLP: A Survey",
            "abstract": "This survey provides an overview of deep learning methods in NLP.",
            "claims": ["Deep learning has revolutionized NLP", "Attention mechanisms are crucial for modern NLP"],
            "year": 2022
        },
        {
            "title": "Efficient Natural Language Processing",
            "abstract": "We present methods for making NLP models more efficient.",
            "claims": ["Model compression can reduce resource usage", "Knowledge distillation improves efficiency"],
            "year": 2023
        }
    ]
    
    # Sample paper content
    sample_paper = """
    Title: A Novel Approach to Machine Learning for Natural Language Processing
    
    Abstract:
    This paper introduces a novel approach to machine learning for natural language processing tasks.
    We demonstrate that our method achieves state-of-the-art performance on several benchmark datasets
    while requiring significantly less computational resources than previous approaches. Our results
    show a 15% improvement in accuracy over the baseline methods. We also present theoretical analysis
    that explains why our approach is more efficient.
    
    Introduction:
    Natural language processing (NLP) has seen remarkable progress in recent years, largely due to
    advances in deep learning. However, these advances have come at the cost of increased computational
    requirements. In this paper, we propose a more efficient approach that maintains high performance
    while reducing resource usage.
    
    Methodology:
    Our approach combines traditional NLP techniques with modern deep learning architectures in a
    novel way. We introduce a new attention mechanism that focuses on the most relevant parts of the
    input text, reducing the need for processing the entire sequence. This allows our model to achieve
    comparable results with fewer parameters and less computation.
    
    Results:
    We evaluated our approach on three standard NLP benchmarks: sentiment analysis, named entity
    recognition, and machine translation. Our method achieved state-of-the-art results on all three
    tasks, with a 15% improvement in accuracy over the previous best methods. Additionally, our
    approach required 40% less training time and 30% less memory during inference.
    
    Conclusion:
    We have presented a novel approach to NLP that achieves superior performance while being more
    computationally efficient. Our results demonstrate that it is possible to improve both accuracy
    and efficiency in NLP systems. Future work will explore applications to other domains and
    further optimizations.
    """
    
    print("\nEvaluating sample paper...")
    results = evaluator.evaluate_paper(sample_paper, training_papers)
    
    print("\nEvaluation Results:")
    print(f"Originality Score: {results['originality_score']:.2f}")
    print(f"Plagiarism Flag: {results['plagiarism_flag']}")
    print(f"Decision: {results['decision']}")
    print(f"Reason: {results['reason']}")
    print(f"Recommended Conference: {results['recommended_conference']}")
    
    print("\nBest Reasoning Path:")
    print(results['best_reasoning_path'])
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_paper_evaluator() 
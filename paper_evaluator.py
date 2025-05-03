import json
import random
import datetime
from typing import Dict, List, Tuple, Any, Optional
from dotenv import load_dotenv
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import re
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import PyPDF2
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import concurrent.futures
import functools
from sentence_transformers import CrossEncoder

# Load environment variables
load_dotenv()

# Global model cache to avoid reloading models
MODEL_CACHE = {
    "sentence_transformer": None,
    "local_model": None,
    "local_tokenizer": None,
    "tfidf_vectorizer": None,
    "dense_retriever": None,  # Pre-trained model for dense retrieval
    "dense_tokenizer": None,
    "cross_encoder": None,
    "retriever_cache": {}  # Cache for retrieval results
}

# JSON validation patterns
JSON_PATTERNS = {
    "claims": r'\{[^{]*"claims":\s*\[(.*?)\][^}]*\}',
    "redundancy": r'\{[^{]*"is_redundant":\s*(true|false)[^}]*"similar_paper_title":\s*"([^"]*)"[^}]*"similarity_score":\s*(\d+(?:\.\d+)?)[^}]*\}',
    "reasoning": r'\{[^{]*"reasoning":\s*"([^"]*)"[^}]*"score":\s*(\d+(?:\.\d+)?)[^}]*\}',
    "evaluation": r'\{[^{]*"originality_score":\s*(\d+(?:\.\d+)?)[^}]*"plagiarism_flag":\s*(true|false)[^}]*"decision":\s*"([^"]*)"[^}]*"recommended_conference":\s*"([^"]*)"[^}]*\}',
    "claim_strength": r'\{[^{]*"claim_strength":\s*(\d+(?:\.\d+)?)[^}]*"reasoning":\s*"([^"]*)"[^}]*\}'
}

# Structured prompts
PROMPTS = {
    "extract_claims": """
    Extract the core claims from this research paper. Return ONLY a JSON object with the following structure:
    {
        "claims": [
            {
                "claim": "string describing the claim",
                "importance": "high/medium/low",
                "novelty": "high/medium/low",
                "evidence": "string describing supporting evidence",
                "impact": "string describing potential impact"
            }
        ]
    }
    """,
    
    "check_redundancy": """
    Check if this claim is redundant with previously published papers. Return ONLY a JSON object with the following structure:
    {
        "is_redundant": boolean,
        "similar_paper_title": "string or null if not redundant",
        "similarity_score": number between 0 and 100
    }
    """,
    
    "reasoning": """
    Evaluate this aspect of the paper. Return ONLY a JSON object with the following structure:
    {
        "reasoning": "string explaining the reasoning",
        "score": number between 0 and 100
    }
    """,
    
    "final_evaluation": """
    Provide the final evaluation of the paper. Return ONLY a JSON object with the following structure:
    {
        "originality_score": number between 0 and 100,
        "plagiarism_flag": boolean,
        "decision": "Accept/Reject/Needs Revision",
        "recommended_conference": "NeurIPS/ICML/ICLR/AAAI/ACL/N/A"
    }
    """,
    
    "claim_strength": """
    Evaluate the strength of this claim. Return ONLY a JSON object with the following structure:
    {
        "claim_strength": number between 0 and 100,
        "reasoning": "string explaining the strength score"
    }
    
    Consider:
    - Clarity and specificity of the claim
    - Quality and quantity of supporting evidence
    - Novelty and significance
    - Potential impact on the field
    """
}

def validate_json_output(text: str, pattern_key: str) -> Optional[Dict]:
    """Validate and extract JSON from LLM output using regex patterns."""
    pattern = JSON_PATTERNS[pattern_key]
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    
    try:
        if pattern_key == "claims":
            # Extract claims array and parse each claim
            claims_text = match.group(1)
            claims = []
            for claim_match in re.finditer(r'\{[^}]*\}', claims_text):
                claim = json.loads(claim_match.group(0))
                claims.append(claim)
            return {"claims": claims}
        
        elif pattern_key == "redundancy":
            return {
                "is_redundant": match.group(1).lower() == "true",
                "similar_paper_title": match.group(2),
                "similarity_score": float(match.group(3))
            }
        
        elif pattern_key == "reasoning":
            return {
                "reasoning": match.group(1),
                "score": float(match.group(2))
            }
        
        elif pattern_key == "evaluation":
            return {
                "originality_score": float(match.group(1)),
                "plagiarism_flag": match.group(2).lower() == "true",
                "decision": match.group(3),
                "recommended_conference": match.group(4)
            }
        
        elif pattern_key == "claim_strength":
            return {
                "claim_strength": float(match.group(1)),
                "reasoning": match.group(2)
            }
    except (json.JSONDecodeError, IndexError, ValueError):
        return None
    
    return None

def calculate_claim_similarity(claim1: str, claim2: str) -> float:
    """
    Calculate similarity between two claims using ColBERT-style retrieval.
    
    Args:
        claim1: First claim to compare
        claim2: Second claim to compare
        
    Returns:
        Similarity score between 0 and 1
    """
    # Check cache first
    cache_key = f"similarity_{hash(claim1)}_{hash(claim2)}"
    if cache_key in MODEL_CACHE["retriever_cache"]:
        return MODEL_CACHE["retriever_cache"][cache_key]
    
    try:
        # Use ColBERT if available
        if MODEL_CACHE["colbert"] is not None and MODEL_CACHE["colbert_index"] is not None:
            # Tokenize claims
            searcher = MODEL_CACHE["colbert"]
            
            tokens1 = searcher.tokenizer(claim1, padding=True, truncation=True, max_length=512, return_tensors="pt")
            tokens2 = searcher.tokenizer(claim2, padding=True, truncation=True, max_length=512, return_tensors="pt")
            
            # Get embeddings
            with torch.no_grad():
                emb1 = searcher.model(**tokens1).last_hidden_state.mean(dim=1)
                emb2 = searcher.model(**tokens2).last_hidden_state.mean(dim=1)
            
            # Calculate similarity using dot product (ColBERT style)
            similarity = torch.mm(emb1, emb2.t()).squeeze().cpu().numpy()
            
            # Cache result
            MODEL_CACHE["retriever_cache"][cache_key] = float(similarity)
            return float(similarity)
        
        # Fallback to sentence transformer if available
        elif MODEL_CACHE["sentence_transformer"] is not None:
            model = MODEL_CACHE["sentence_transformer"]
            
            # Encode claims
            emb1 = model.encode(claim1, convert_to_tensor=True)
            emb2 = model.encode(claim2, convert_to_tensor=True)
            
            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).cpu().numpy()
            
            # Cache result
            MODEL_CACHE["retriever_cache"][cache_key] = float(similarity)
            return float(similarity)
        
        # Last resort: use TF-IDF
        else:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([claim1, claim2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Cache result
            MODEL_CACHE["retriever_cache"][cache_key] = float(similarity)
            return float(similarity)
            
    except Exception as e:
        print(f"Error calculating claim similarity: {e}")
        # Return a default similarity score
        return 0.5

def deduplicate_claims(claims: List[Dict], similarity_threshold: float = 0.7) -> List[Dict]:
    """Deduplicate claims based on semantic similarity."""
    if not claims:
        return []
    
    # Sort claims by importance and novelty (high to low)
    importance_score = {"high": 3, "medium": 2, "low": 1}
    novelty_score = {"high": 3, "medium": 2, "low": 1}
    
    claims.sort(key=lambda x: (
        importance_score.get(x.get("importance", "low"), 1) +
        novelty_score.get(x.get("novelty", "low"), 1)
    ), reverse=True)
    
    # Initialize with the first claim
    unique_claims = [claims[0]]
    
    # Check each remaining claim against unique claims
    for claim in claims[1:]:
        is_duplicate = False
        for unique_claim in unique_claims:
            similarity = calculate_claim_similarity(claim["claim"], unique_claim["claim"])
            if similarity > similarity_threshold:
                # If duplicate found, merge evidence and impact if they're different
                if claim.get("evidence") and claim["evidence"] != unique_claim.get("evidence"):
                    unique_claim["evidence"] = f"{unique_claim.get('evidence', '')} | {claim['evidence']}"
                if claim.get("impact") and claim["impact"] != unique_claim.get("impact"):
                    unique_claim["impact"] = f"{unique_claim.get('impact', '')} | {claim['impact']}"
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_claims.append(claim)
    
    return unique_claims

def calculate_claim_strength(claim: Dict, model_type: str) -> float:
    """Calculate the strength score of a claim using LLM."""
    prompt = PROMPTS["claim_strength"] + f"\n\nClaim: {claim['claim']}\nEvidence: {claim.get('evidence', 'N/A')}\nImpact: {claim.get('impact', 'N/A')}"
    response = call_free_model_api(prompt, model_type=model_type, temperature=0.3)
    
    result = validate_json_output(response, "claim_strength")
    if not result:
        # Fallback: Calculate based on importance and novelty
        importance_score = {"high": 100, "medium": 70, "low": 40}
        novelty_score = {"high": 100, "medium": 70, "low": 40}
        
        base_score = (
            importance_score.get(claim.get("importance", "low"), 40) +
            novelty_score.get(claim.get("novelty", "low"), 40)
        ) / 2
        
        # Adjust based on evidence and impact
        if claim.get("evidence"):
            base_score += 10
        if claim.get("impact"):
            base_score += 10
        
        return min(100, base_score)
    
    return result["claim_strength"]

def parallel_process_claims(claims: List[Dict], model_type: str, max_workers: int = 4) -> List[Dict]:
    """
    Process claims in parallel to improve performance.
    
    Args:
        claims: List of claims to process
        model_type: Type of model to use
        max_workers: Maximum number of parallel workers
        
    Returns:
        Processed claims with strength scores
    """
    # Create a partial function with fixed model_type
    process_claim = functools.partial(calculate_claim_strength, model_type=model_type)
    
    # Process claims in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_claim = {executor.submit(process_claim, claim): claim for claim in claims}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_claim):
            claim = future_to_claim[future]
            try:
                strength_score = future.result()
                claim["strength_score"] = strength_score
            except Exception as e:
                print(f"Error processing claim: {e}")
                # Fallback score
                claim["strength_score"] = 50.0
    
    return claims

def parallel_check_redundancy(claims: List[Dict], similar_papers: List[Dict], max_workers: int = 4) -> List[Dict]:
    """
    Check redundancy of claims in parallel.
    
    Args:
        claims: List of claims to check
        similar_papers: List of similar papers
        max_workers: Maximum number of parallel workers
        
    Returns:
        Claims with redundancy information
    """
    def check_single_claim(claim):
        claim_text = claim["claim"]
        is_redundant, similar_paper, similarity = check_redundancy(claim_text, similar_papers)
        
        claim["is_redundant"] = is_redundant
        claim["similar_paper"] = similar_paper
        claim["similarity"] = similarity
        
        return claim
    
    # Process claims in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_claim = {executor.submit(check_single_claim, claim): claim for claim in claims}
        
        # Process results as they complete
        processed_claims = []
        for future in concurrent.futures.as_completed(future_to_claim):
            try:
                processed_claim = future.result()
                processed_claims.append(processed_claim)
            except Exception as e:
                print(f"Error checking redundancy: {e}")
                # Add claim with default values
                claim = future_to_claim[future]
                claim["is_redundant"] = False
                claim["similar_paper"] = ""
                claim["similarity"] = 0.0
                processed_claims.append(claim)
    
    return processed_claims

def check_redundancy(claim: str, similar_papers: List[Dict]) -> Tuple[bool, str, float]:
    """
    Check if a claim is redundant with previously published papers.
    
    Args:
        claim: The claim to check
        similar_papers: List of similar papers
        
    Returns:
        Tuple of (is_redundant, similar_paper_title, similarity_score)
    """
    if not similar_papers:
        return False, "", 0.0
    
    # Find the most similar paper
    most_similar = max(similar_papers, key=lambda x: x["similarity_score"])
    
    # Check if the similarity is above threshold
    similarity_threshold = 0.7
    is_redundant = most_similar["similarity_score"] > similarity_threshold
    
    return is_redundant, most_similar["title"], most_similar["similarity_score"]

def call_free_model_api(prompt: str, model_type: str = "simulated", temperature: float = 0.7) -> str:
    """
    Call a free model API to get a response.
    
    Args:
        prompt: The prompt to send to the model
        model_type: The type of model to use ("huggingface", "local", or "simulated")
        temperature: The temperature for generation (0.0 to 1.0)
        
    Returns:
        The model's response as a string
    """
    if model_type == "huggingface":
        # Use Hugging Face's free inference API
        try:
            # Use a free model from Hugging Face
            model_id = "mistralai/Mistral-7B-Instruct-v0.2"
            
            # Check if we need to load the model
            if MODEL_CACHE["local_model"] is None:
                print(f"Loading model {model_id}...")
                MODEL_CACHE["local_tokenizer"] = AutoTokenizer.from_pretrained(model_id)
                MODEL_CACHE["local_model"] = AutoModelForCausalLM.from_pretrained(
                    model_id, 
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                print("Model loaded successfully.")
            
            # Generate response
            inputs = MODEL_CACHE["local_tokenizer"](prompt, return_tensors="pt").to(MODEL_CACHE["local_model"].device)
            outputs = MODEL_CACHE["local_model"].generate(
                **inputs,
                max_length=1024,
                temperature=temperature,
                do_sample=True,
                top_p=0.95
            )
            response = MODEL_CACHE["local_tokenizer"].decode(outputs[0], skip_special_tokens=True)
            
            # Extract the response part (after the prompt)
            response = response[len(prompt):].strip()
            return response
            
        except Exception as e:
            print(f"Error calling Hugging Face model: {e}")
            print("Falling back to simulated model.")
            return simulate_model_response(prompt, temperature)
    
    elif model_type == "local":
        # Use a locally loaded model
        try:
            # Check if we need to load the model
            if MODEL_CACHE["local_model"] is None:
                print("Loading local model...")
                model_id = "mistralai/Mistral-7B-Instruct-v0.2"
                MODEL_CACHE["local_tokenizer"] = AutoTokenizer.from_pretrained(model_id)
                MODEL_CACHE["local_model"] = AutoModelForCausalLM.from_pretrained(
                    model_id, 
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                print("Local model loaded successfully.")
            
            # Generate response
            inputs = MODEL_CACHE["local_tokenizer"](prompt, return_tensors="pt").to(MODEL_CACHE["local_model"].device)
            outputs = MODEL_CACHE["local_model"].generate(
                **inputs,
                max_length=1024,
                temperature=temperature,
                do_sample=True,
                top_p=0.95
            )
            response = MODEL_CACHE["local_tokenizer"].decode(outputs[0], skip_special_tokens=True)
            
            # Extract the response part (after the prompt)
            response = response[len(prompt):].strip()
            return response
            
        except Exception as e:
            print(f"Error calling local model: {e}")
            print("Falling back to simulated model.")
            return simulate_model_response(prompt, temperature)
    
    else:  # "simulated" or any other value
        # Use a simulated model for testing
        return simulate_model_response(prompt, temperature)

def simulate_model_response(prompt: str, temperature: float = 0.7) -> str:
    """
    Simulate a model response for testing purposes.
    
    Args:
        prompt: The prompt to respond to
        temperature: The temperature for generation (0.0 to 1.0)
        
    Returns:
        A simulated response
    """
    # Add a small delay to simulate API call
    time.sleep(0.5)
    
    # Check if the prompt is asking for claims
    if "claims" in prompt.lower() and "extract" in prompt.lower():
        return """
        {
            "claims": [
                {
                    "claim": "This paper introduces a novel approach to machine learning",
                    "importance": "high",
                    "novelty": "high",
                    "evidence": "Experimental results show 15% improvement over baseline",
                    "impact": "Could revolutionize the field of AI"
                },
                {
                    "claim": "The proposed method is computationally efficient",
                    "importance": "medium",
                    "novelty": "medium",
                    "evidence": "Runtime analysis shows O(n log n) complexity",
                    "impact": "Enables deployment on resource-constrained devices"
                }
            ]
        }
        """
    
    # Check if the prompt is asking for claim strength
    elif "claim_strength" in prompt.lower():
        return """
        {
            "claim_strength": 85,
            "reasoning": "The claim is well-supported by experimental evidence and has significant potential impact."
        }
        """
    
    # Check if the prompt is asking for redundancy
    elif "redundant" in prompt.lower():
        return """
        {
            "is_redundant": false,
            "similar_paper_title": "A Related Paper from 2022",
            "similarity_score": 0.35
        }
        """
    
    # Check if the prompt is asking for reasoning
    elif "reasoning" in prompt.lower() and "score" in prompt.lower():
        return """
        {
            "reasoning": "The paper presents a novel approach with strong experimental validation.",
            "score": 0.85
        }
        """
    
    # Check if the prompt is asking for final evaluation
    elif "final_evaluation" in prompt.lower() or "originality_score" in prompt.lower():
        return """
        {
            "originality_score": 0.85,
            "plagiarism_flag": false,
            "decision": "Accept",
            "recommended_conference": "NeurIPS"
        }
        """
    
    # Default response
    return "This is a simulated response to your prompt. In a real implementation, this would be generated by a language model."

class PaperEvaluator:
    """
    A system to evaluate research papers for publishability and recommend conferences.
    """
    
    def __init__(self, model_type: str = "simulated", max_workers: int = 4):
        """
        Initialize the paper evaluator.
        
        Args:
            model_type: The type of model to use ("huggingface", "local", or "simulated")
            max_workers: Maximum number of parallel workers for processing
        """
        self.conferences = ["NeurIPS", "ICML", "ICLR", "AAAI", "ACL"]
        self.model_type = model_type
        self.max_workers = max_workers
        
        # Initialize sentence transformer as fallback
        try:
            if MODEL_CACHE["sentence_transformer"] is None:
                print("Loading sentence transformer model...")
                MODEL_CACHE["sentence_transformer"] = SentenceTransformer('msmarco-distilbert-base-v3')
                print("Sentence transformer model loaded successfully.")
            
            self.sentence_transformer = MODEL_CACHE["sentence_transformer"]
            self.embedding_model_available = True
        except Exception as e:
            print(f"Warning: Could not load sentence transformer model: {e}")
            print("Falling back to TF-IDF for similarity calculations.")
            self.embedding_model_available = False
            
            # Initialize TF-IDF vectorizer in the global cache
            if MODEL_CACHE["tfidf_vectorizer"] is None:
                MODEL_CACHE["tfidf_vectorizer"] = TfidfVectorizer()

    def dense_retrieve(self, query: str, training_papers: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Retrieve similar papers using a pre-trained dense retriever.
        
        Args:
            query: The query text (paper content or claim)
            training_papers: List of papers to search in
            top_k: Number of papers to retrieve
            
        Returns:
            List of similar papers with their metadata and similarity scores
        """
        # Check cache first
        cache_key = f"{query[:100]}_{len(training_papers)}"  # Use first 100 chars as key
        if cache_key in MODEL_CACHE["retriever_cache"]:
            return MODEL_CACHE["retriever_cache"][cache_key]
        
        if not training_papers:
            return []
        
        try:
            # Initialize model if not already done
            if MODEL_CACHE["dense_retriever"] is None:
                print("Loading dense retriever model...")
                MODEL_CACHE["dense_retriever"] = AutoModel.from_pretrained("sentence-transformers/msmarco-distilbert-base-v3")
                MODEL_CACHE["dense_tokenizer"] = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-base-v3")
                print("Dense retriever model loaded successfully.")
            
            # Prepare papers for retrieval
            paper_texts = []
            for paper in training_papers:
                paper_text = f"{paper['title']} {paper['abstract']} {' '.join(paper['claims'])}"
                paper_texts.append(paper_text)
            
            # Tokenize query and papers
            query_tokens = MODEL_CACHE["dense_tokenizer"](query, padding=True, truncation=True, max_length=512, return_tensors="pt")
            paper_tokens = MODEL_CACHE["dense_tokenizer"](paper_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            
            # Get embeddings
            with torch.no_grad():
                query_embeddings = MODEL_CACHE["dense_retriever"](**query_tokens).last_hidden_state.mean(dim=1)
                paper_embeddings = MODEL_CACHE["dense_retriever"](**paper_tokens).last_hidden_state.mean(dim=1)
            
            # Calculate similarities using dot product
            similarities = torch.mm(query_embeddings, paper_embeddings.t()).squeeze().cpu().numpy()
            
            # Get top-k papers
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            results = []
            
            for idx in top_indices:
                paper_data = training_papers[idx].copy()
                paper_data["similarity_score"] = float(similarities[idx])
                results.append(paper_data)
            
            # Cache results
            MODEL_CACHE["retriever_cache"][cache_key] = results
            return results
            
        except Exception as e:
            print(f"Error in dense retrieval: {e}")
            # Fallback to sentence transformer
            return self.sentence_transformer_retrieve(query, training_papers, top_k)

    def sentence_transformer_retrieve(self, query: str, training_papers: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Retrieve similar papers using sentence transformer as fallback.
        
        Args:
            query: The query text (paper content or claim)
            training_papers: List of papers to search in
            top_k: Number of papers to retrieve
            
        Returns:
            List of similar papers with their metadata and similarity scores
        """
        # Check cache first
        cache_key = f"st_{query[:100]}_{len(training_papers)}"  # Use first 100 chars as key
        if cache_key in MODEL_CACHE["retriever_cache"]:
            return MODEL_CACHE["retriever_cache"][cache_key]
        
        if not training_papers:
            return []
        
        # Prepare papers for retrieval
        paper_texts = []
        for paper in training_papers:
            # Combine title, abstract, and claims for better matching
            paper_text = f"{paper['title']} {paper['abstract']} {' '.join(paper['claims'])}"
            paper_texts.append(paper_text)
        
        try:
            # Encode query and papers
            query_embedding = self.sentence_transformer.encode(query, convert_to_tensor=True)
            paper_embeddings = self.sentence_transformer.encode(paper_texts, convert_to_tensor=True)
            
            # Calculate similarities
            similarities = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0), paper_embeddings
            ).cpu().numpy()
            
            # Get top-k papers
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            results = []
            
            for idx in top_indices:
                paper_data = training_papers[idx].copy()
                paper_data["similarity_score"] = float(similarities[idx])
                results.append(paper_data)
            
            # Cache results
            MODEL_CACHE["retriever_cache"][cache_key] = results
            return results
            
        except Exception as e:
            print(f"Error in sentence transformer retrieval: {e}")
            return []

    def retrieve_similar_papers(self, claims: List[str], training_papers: List[Dict] = None) -> List[Dict]:
        """
        Retrieve similar papers based on semantic similarity of claims.
        
        Args:
            claims: List of claims from the input paper
            training_papers: List of papers to search for similarity
            
        Returns:
            List of similar papers with their metadata, sorted by similarity
        """
        if not claims or not training_papers:
            return []
        
        # Use dense retrieval for each claim
        all_similar_papers = []
        for claim in claims:
            similar_papers = self.dense_retrieve(claim, training_papers, top_k=3)
            all_similar_papers.extend(similar_papers)
        
        # Remove duplicates and sort by similarity
        unique_papers = {}
        for paper in all_similar_papers:
            title = paper["title"]
            if title not in unique_papers or paper["similarity_score"] > unique_papers[title]["similarity_score"]:
                unique_papers[title] = paper
        
        # Convert back to list and sort by similarity
        result = list(unique_papers.values())
        result.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return result
    
    def check_redundancy(self, claim: str, similar_papers: List[Dict]) -> Tuple[bool, str, float]:
        """
        Check if a claim is redundant with previously published papers.
        
        Args:
            claim: The claim to check
            similar_papers: List of similar papers
            
        Returns:
            Tuple of (is_redundant, similar_paper_title, similarity_score)
        """
        if not similar_papers:
            return False, "", 0.0
        
        # Find the most similar paper
        most_similar = max(similar_papers, key=lambda x: x["similarity_score"])
        
        # Check if the similarity is above threshold
        similarity_threshold = 0.7
        is_redundant = most_similar["similarity_score"] > similarity_threshold
        
        return is_redundant, most_similar["title"], most_similar["similarity_score"]
    
    def extract_claims(self, paper_content: str, training_papers: List[Dict] = None) -> List[Dict]:
        """Extract core claims from the paper using GoT approach."""
        # Initialize empty training papers if none provided
        if training_papers is None:
            training_papers = []
        
        # First retrieve similar papers for context (only if we have training papers)
        similar_papers = []
        if training_papers:
            similar_papers = self.dense_retrieve(paper_content[:1000], training_papers, top_k=3)
        
        # Prepare context with similar papers
        context = {
            "paper_content": paper_content,
            "similar_papers": [
                {
                    "title": p["title"],
                    "abstract": p["abstract"],
                    "claims": p["claims"],
                    "year": p.get("year", "N/A")
                } for p in similar_papers
            ]
        }
        
        prompt = PROMPTS["extract_claims"] + "\n\nContext:\n" + json.dumps(context, indent=2)
        response = call_free_model_api(prompt, model_type="mistral", temperature=0.3)
        
        result = validate_json_output(response, "claims")
        if not result or "claims" not in result:
            print("Warning: Failed to extract claims properly. Using fallback extraction.")
            # Fallback: Simple extraction of sentences that look like claims
            claims = []
            sentences = paper_content.split('.')
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in ["propose", "demonstrate", "show", "introduce", "present"]):
                    claims.append({
                        "claim": sentence.strip(),
                        "importance": "medium",
                        "novelty": "medium",
                        "evidence": "",
                        "impact": ""
                    })
            claims = claims[:5]  # Limit to 5 claims as fallback
        else:
            claims = result["claims"]
        
        # Deduplicate claims
        unique_claims = deduplicate_claims(claims)
        
        # Calculate strength score for each claim in parallel
        unique_claims = parallel_process_claims(unique_claims, "mistral", self.max_workers)
        
        return unique_claims

    def graph_of_thoughts_filter(self, paper_content: str, training_papers: List[Dict] = None) -> Tuple[bool, List[Dict], float]:
        """Apply Graph of Thought filter to check for redundant claims."""
        if not training_papers:
            print("Warning: Graph of Thoughts filter running without training papers.")
            print("All claims will be considered non-redundant by default.")
        
        # Extract and deduplicate claims
        claims = self.extract_claims(paper_content, training_papers)
        
        # Retrieve similar papers for each claim
        similar_papers = self.retrieve_similar_papers([c["claim"] for c in claims], training_papers)
        
        # Check redundancy for all claims in parallel
        processed_claims = parallel_check_redundancy(claims, similar_papers, self.max_workers)
        
        # Analyze results
        redundant_count = sum(1 for claim in processed_claims if claim["is_redundant"])
        total_strength = sum(claim["strength_score"] for claim in processed_claims)
        
        # Calculate weighted originality score (0-100)
        if processed_claims:
            avg_strength = total_strength / len(processed_claims)
            originality_score = avg_strength * (1 - (redundant_count / len(processed_claims)))
        else:
            originality_score = 0
        
        # Paper passes if less than 70% of claims are redundant
        passes_filter = redundant_count < len(processed_claims) * 0.7
        
        return passes_filter, processed_claims, originality_score

    def generate_reasoning_path(self, prompt: str, context: Dict) -> str:
        """
        Generate a reasoning path using the Actor LLM.
        
        Args:
            prompt: The prompt for the reasoning
            context: Context information
            
        Returns:
            Generated reasoning path
        """
        full_prompt = f"""
        {prompt}
        
        Context:
        {json.dumps(context, indent=2)}
        """
        
        return call_free_model_api(full_prompt, model_type=self.model_type, temperature=0.7)
    
    def evaluate_reasoning_path(self, reasoning: str, context: Dict) -> float:
        """
        Evaluate a reasoning path using the Critic LLM.
        
        Args:
            reasoning: The reasoning path to evaluate
            context: Context information
            
        Returns:
            Score between 0 and 1
        """
        prompt = f"""
        Evaluate this reasoning path on a scale of 0 to 1, considering:
        - Clarity: Is the reasoning clear and well-structured?
        - Novelty: Does it offer novel insights?
        - Relevance: Is it relevant to the paper evaluation?
        
        Return only a number between 0 and 1.
        
        Reasoning:
        {reasoning}
        
        Context:
        {json.dumps(context, indent=2)}
        """
        
        response = call_free_model_api(prompt, model_type=self.model_type, temperature=0.3)
        
        try:
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except ValueError:
            return 0.5  # Default score if parsing fails
    
    def tree_of_thoughts_reasoning(self, paper_content: str, claim_analysis: List[Dict]) -> Dict:
        """
        Apply Tree of Thought reasoning to evaluate the paper.
        
        Args:
            paper_content: The content of the paper
            claim_analysis: Analysis of claims from the GoT filter
            
        Returns:
            Dictionary with reasoning results
        """
        # Retrieve similar papers for context
        similar_papers = self.dense_retrieve(paper_content[:1000], self.training_papers, top_k=5)
        
        context = {
            "paper_content": paper_content[:1000],  # Limiting content length
            "claim_analysis": claim_analysis,
            "similar_papers": [
                {
                    "title": p["title"],
                    "abstract": p["abstract"],
                    "claims": p["claims"],
                    "year": p.get("year", "N/A"),
                    "similarity": p["similarity_score"]
                } for p in similar_papers
            ]
        }
        
        # Initialize the tree with the root node
        root_prompt = "Evaluate this research paper for publishability. Consider originality, methodology, significance, and potential impact."
        root_reasoning = self.generate_reasoning_path(root_prompt, context)
        root_score = self.evaluate_reasoning_path(root_reasoning, context)
        
        # First level of reasoning (3 branches)
        level1_prompts = [
            "Focus on the originality and novelty of the research.",
            "Focus on the methodology and technical soundness.",
            "Focus on the significance and potential impact of the research."
        ]
        
        level1_reasonings = []
        level1_scores = []
        
        for prompt in level1_prompts:
            reasoning = self.generate_reasoning_path(prompt, context)
            score = self.evaluate_reasoning_path(reasoning, context)
            level1_reasonings.append(reasoning)
            level1_scores.append(score)
        
        # Keep top 2 branches
        level1_indices = sorted(range(len(level1_scores)), key=lambda i: level1_scores[i], reverse=True)[:2]
        level1_selected = [(level1_reasonings[i], level1_scores[i]) for i in level1_indices]
        
        # Second level of reasoning (2 branches per selected first-level branch)
        level2_reasonings = []
        level2_scores = []
        
        for reasoning, score in level1_selected:
            level2_prompts = [
                f"Build on this reasoning: {reasoning[:100]}... Consider strengths and weaknesses.",
                f"Build on this reasoning: {reasoning[:100]}... Consider practical applications and limitations."
            ]
            
            for prompt in level2_prompts:
                new_reasoning = self.generate_reasoning_path(prompt, context)
                new_score = self.evaluate_reasoning_path(new_reasoning, context)
                level2_reasonings.append(new_reasoning)
                level2_scores.append(new_score)
        
        # Keep top 2 branches
        level2_indices = sorted(range(len(level2_scores)), key=lambda i: level2_scores[i], reverse=True)[:2]
        level2_selected = [(level2_reasonings[i], level2_scores[i]) for i in level2_indices]
        
        # Third level of reasoning (2 branches per selected second-level branch)
        level3_reasonings = []
        level3_scores = []
        
        for reasoning, score in level2_selected:
            level3_prompts = [
                f"Build on this reasoning: {reasoning[:100]}... Make a final assessment of publishability.",
                f"Build on this reasoning: {reasoning[:100]}... Recommend the most suitable conference."
            ]
            
            for prompt in level3_prompts:
                new_reasoning = self.generate_reasoning_path(prompt, context)
                new_score = self.evaluate_reasoning_path(new_reasoning, context)
                level3_reasonings.append(new_reasoning)
                level3_scores.append(new_score)
        
        # Find the best reasoning path
        best_index = max(range(len(level3_scores)), key=lambda i: level3_scores[i])
        best_reasoning = level3_reasonings[best_index]
        
        return {
            "root_reasoning": root_reasoning,
            "root_score": root_score,
            "level1_reasonings": level1_reasonings,
            "level1_scores": level1_scores,
            "level1_selected": level1_selected,
            "level2_reasonings": level2_reasonings,
            "level2_scores": level2_scores,
            "level2_selected": level2_selected,
            "level3_reasonings": level3_reasonings,
            "level3_scores": level3_scores,
            "best_reasoning": best_reasoning,
            "best_score": level3_scores[best_index]
        }
    
    def make_final_evaluation(self, paper_content: str, claim_analysis: List[Dict], tot_results: Dict) -> Dict:
        """
        Make the final evaluation of the paper.
        
        Args:
            paper_content: The content of the paper
            claim_analysis: Analysis of claims from the GoT filter
            tot_results: Results from the Tree of Thought reasoning
            
        Returns:
            Dictionary with final evaluation results
        """
        # Count redundant claims
        redundant_count = sum(1 for claim in claim_analysis if claim["is_redundant"])
        plagiarism_flag = redundant_count > 0
        
        # Determine if paper passes based on GoT filter and ToT reasoning
        passes_got = redundant_count < len(claim_analysis) * 0.7
        passes_tot = tot_results["best_score"] > 0.6
        
        # Make final decision
        if not passes_got:
            decision = "Reject"
            reason = "Too many redundant claims detected"
        elif not passes_tot:
            decision = "Needs Revision"
            reason = "Paper has potential but needs significant improvements"
        else:
            decision = "Accept"
            reason = "Paper meets quality standards for publication"
        
        # Recommend conference
        conference_prompt = f"""
        Based on the paper content and analysis, recommend the most suitable conference among: {', '.join(self.conferences)}.
        Consider the paper's topic, methodology, and contribution.
        
        Paper content (excerpt):
        {paper_content[:500]}
        
        Analysis:
        {tot_results["best_reasoning"]}
        
        Return only the conference name.
        """
        
        conference_response = call_free_model_api(conference_prompt, model_type=self.model_type, temperature=0.3)
        recommended_conference = conference_response.strip()
        
        # Ensure the recommended conference is in our list
        if recommended_conference not in self.conferences:
            recommended_conference = random.choice(self.conferences)
        
        return {
            "originality_score": 100 * (1 - (redundant_count / len(claim_analysis))) if claim_analysis else 0,
            "plagiarism_flag": "Yes" if plagiarism_flag else "No",
            "best_reasoning_path": tot_results["best_reasoning"],
            "decision": decision,
            "reason": reason,
            "recommended_conference": recommended_conference
        }
    
    def evaluate_paper(self, paper_content: str, training_papers: List[Dict] = None) -> Dict:
        """
        Evaluate a research paper using Graph of Thoughts and Tree of Thoughts approaches.
        
        Args:
            paper_content: The content of the paper
            training_papers: List of papers to use for similarity comparison
            
        Returns:
            Dictionary with evaluation results
        """
        if not training_papers:
            print("\nWarning: No training papers provided. The evaluation will proceed without redundancy checking.")
            print("This may result in less accurate evaluation as claims cannot be compared against existing work.\n")
        
        # Apply Graph of Thoughts filter
        passes_got, claim_analysis, got_score = self.graph_of_thoughts_filter(paper_content, training_papers)
        
        # If paper fails the GoT filter, return early with rejection
        if not passes_got:
            return {
                "originality_score": got_score,
                "plagiarism_flag": "Yes",
                "best_reasoning_path": "Paper rejected due to redundant claims.",
                "decision": "Reject",
                "recommended_conference": "N/A"
            }
        
        # Step 2: Apply Tree of Thought reasoning
        tot_results = self.tree_of_thoughts_reasoning(paper_content, claim_analysis)
        
        # Step 3: Make final evaluation
        final_evaluation = self.make_final_evaluation(paper_content, claim_analysis, tot_results)
        
        return final_evaluation

def main():
    """Main function to demonstrate the paper evaluator."""
    # Get model type from command line or use default
    import sys
    model_type = "simulated"  # Default
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
        if model_type not in ["huggingface", "local", "simulated"]:
            print(f"Unknown model type: {model_type}. Using 'simulated' instead.")
            model_type = "simulated"
    
    print(f"Using model type: {model_type}")
    
    # Create evaluator
    evaluator = PaperEvaluator(model_type=model_type)
    
    # Print usage instructions
    print("\n===== PAPER EVALUATOR DEMO =====")
    print("This module is designed to be imported by pdf_processor.py")
    print("To use this module directly, provide a paper content as a string:")
    print("evaluator = PaperEvaluator(model_type='simulated')")
    print("results = evaluator.evaluate_paper(paper_content)")
    print("\nFor full functionality with PDF processing, use pdf_processor.py")

if __name__ == "__main__":
    main() 
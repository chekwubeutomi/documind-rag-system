"""
Script to evaluate the RAG system using custom evaluation metrics.

This script tests the RAG system's quality by:
1. Loading a set of test questions
2. Querying the RAG system
3. Recording answers and metrics
4. Saving results to JSON file

Evaluation Approach:
- Manual/Human evaluation: Review answers for quality
- Automatic metrics: Processing time, retrieval count, etc.
- Compare with expected answers: Check if system answered correctly

Why Evaluate?
- Measure system quality (did it improve?)
- Identify failing cases (which questions are hard?)
- Performance monitoring (how fast is it?)
- A/B testing (new embedding model vs old?)

Test Questions Format (JSON):
[
    {
        "question": "What is FastAPI?",
        "expected_answer": "FastAPI is a modern web framework...",
        "context": "Optional context or category"
    },
    {
        "question": "How do you install FastAPI?",
        "expected_answer": "pip install fastapi"
    }
]

Usage:
$ python scripts/evaluate_rag.py test_questions.json
$ python scripts/evaluate_rag.py test_questions.json --api-url http://localhost:8000

Output:
- evaluation_results.json: Detailed results with answers and metrics
- Console logs: Real-time progress and summary statistics
"""
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging import logger


def load_test_questions(filepath: str) -> List[Dict]:
    """
    Load test questions from JSON file.
    
    Expected format:
    [
        {
            "question": "What is machine learning?",
            "expected_answer": "Machine learning is...",
            "context": "Optional category or notes"
        },
        ...
    ]
    
    Args:
        filepath (str): Path to JSON file with test questions
    
    Returns:
        List[Dict]: List of test question dictionaries
        Each dict has 'question', 'expected_answer', etc.
    
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is invalid JSON
    
    Example:
        >>> questions = load_test_questions("tests.json")
        >>> print(f"Loaded {len(questions)} questions")
        Loaded 10 questions
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def evaluate_rag(questions_file: str, api_url: str = "http://localhost:8000"):
    """
    Evaluate the RAG system on a set of test questions.
    
    This function is the main evaluation logic:
    1. Load test questions from JSON
    2. For each question, query the RAG API
    3. Record answer and metrics
    4. Save results to JSON
    5. Print summary statistics
    
    Evaluation Metrics Collected:
    - question: The original question
    - answer: What the RAG system answered
    - expected: What the expected answer is (for manual comparison)
    - sources_count: How many documents were retrieved
    - processing_time: How long the query took (seconds)
    
    Error Handling:
    - Failed queries are skipped (logged)
    - API connection errors are caught
    - Results only include successful queries
    
    Args:
        questions_file (str): Path to JSON file with test questions
            Example: "data/test_questions.json"
        
        api_url (str): Base URL of the RAG API. Default "http://localhost:8000"
            Example: "http://api.example.com:8000"
    
    Output Files:
    - evaluation_results.json: Saved in same directory as questions_file
        Contains list of results with all metrics
        Can be imported into analysis tools
    
    Log Output:
    - Progress: "[1/10] Evaluating: What is..."
    - Timing: "✓ Answer generated in 1.23s"
    - Summary: "Questions evaluated: 10/10"
    - Statistics: "Average processing time: 1.45s"
    
    Example Usage:
        >>> evaluate_rag("test_questions.json", "http://localhost:8000")
        INFO Loading test questions from: test_questions.json
        INFO Loaded 10 test questions
        INFO [1/10] Evaluating: What is machine learning?
        INFO ✓ Answer generated in 1.23s
        ...
        INFO Evaluation complete!
        INFO Results saved to: evaluation_results.json
        INFO Average processing time: 1.45s
    """
    # Import requests here (optional dependency for this script)
    import requests
    
    # ======================================================================
    # STEP 1: Load Test Questions
    # ======================================================================
    logger.info(f"Loading test questions from: {questions_file}")
    questions = load_test_questions(questions_file)
    logger.info(f"Loaded {len(questions)} test questions")
    
    # ======================================================================
    # STEP 2: Evaluate Each Question
    # ======================================================================
    results = []  # Will accumulate evaluation results
    
    for i, item in enumerate(questions, 1):  # Start counting from 1
        # Extract question and expected answer
        question = item.get("question")
        expected = item.get("expected_answer", "")  # Default to empty string
        
        logger.info(f"[{i}/{len(questions)}] Evaluating: {question}")
        
        try:
            # ============================================================
            # Query the RAG System
            # ============================================================
            # Make HTTP request to the API
            response = requests.post(
                f"{api_url}/api/v1/query",  # Endpoint URL
                json={"query": question, "top_k": 5},  # Request body
                timeout=30  # Wait max 30 seconds
            )
            
            # Check if request was successful
            if response.status_code != 200:
                logger.error(f"Query failed: {response.status_code}")
                continue  # Skip to next question
            
            # Parse response
            data = response.json()
            
            # ============================================================
            # Record Result
            # ============================================================
            # Collect all metrics for this question
            result = {
                "question": question,  # The question asked
                "answer": data.get("answer", ""),  # System's answer
                "expected": expected,  # Expected answer (for comparison)
                "sources_count": len(data.get("sources", [])),  # How many docs retrieved
                "processing_time": data.get("processing_time", 0)  # Query time (seconds)
            }
            
            results.append(result)
            
            logger.info(f"✓ Answer generated in {result['processing_time']:.2f}s")
        
        except Exception as e:
            # Log error but continue with next question
            logger.error(f"Error evaluating question: {e}")
    
    # ======================================================================
    # STEP 3: Save Results
    # ======================================================================
    # Save results to JSON file for later analysis
    output_file = Path(questions_file).parent / "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)  # Pretty-print with indentation
    
    # ======================================================================
    # STEP 4: Print Summary
    # ======================================================================
    logger.info(f"\nEvaluation complete!")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Questions evaluated: {len(results)}/{len(questions)}")
    
    # Calculate and print statistics
    if results:
        # Average processing time across all queries
        avg_time = sum(r["processing_time"] for r in results) / len(results)
        logger.info(f"Average processing time: {avg_time:.2f}s")
        
        # Optional: More detailed statistics
        times = [r["processing_time"] for r in results]
        logger.info(f"Min processing time: {min(times):.2f}s")
        logger.info(f"Max processing time: {max(times):.2f}s")
        
        # Average retrieval count
        avg_sources = sum(r["sources_count"] for r in results) / len(results)
        logger.info(f"Average sources retrieved: {avg_sources:.1f}")


def main():
    """
    Main entry point for command-line usage.
    
    Parses arguments and calls evaluate_rag().
    
    Usage:
        python scripts/evaluate_rag.py test_questions.json
        python scripts/evaluate_rag.py test_questions.json --api-url http://localhost:8000
    
    Arguments:
        questions_file: Path to JSON file with test questions (required)
        --api-url: API URL (optional, default http://localhost:8000)
    """
    # Create argument parser
    parser = argparse.ArgumentParser(description="Evaluate the RAG system")
    
    # Positional argument: questions file
    parser.add_argument(
        "questions_file",
        help="JSON file with test questions and expected answers"
    )
    
    # Optional argument: API URL
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the API"
    )
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    try:
        # Run evaluation
        evaluate_rag(args.questions_file, args.api_url)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)  # Exit with error code


if __name__ == "__main__":
    main()

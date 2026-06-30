"""
Script to evaluate the RAG system using RAGAS metrics or custom evaluation.
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
            "question": "What is FastAPI?",
            "expected_answer": "FastAPI is a modern Python web framework...",
            "context": "Document context..."
        }
    ]
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def evaluate_rag(questions_file: str, api_url: str = "http://localhost:8000"):
    """
    Evaluate the RAG system on a set of test questions.
    
    Args:
        questions_file: Path to JSON file with test questions
        api_url: Base URL of the API
    """
    import requests
    
    logger.info(f"Loading test questions from: {questions_file}")
    questions = load_test_questions(questions_file)
    
    logger.info(f"Loaded {len(questions)} test questions")
    
    results = []
    
    for i, item in enumerate(questions, 1):
        question = item.get("question")
        expected = item.get("expected_answer", "")
        
        logger.info(f"[{i}/{len(questions)}] Evaluating: {question}")
        
        try:
            # Query the RAG system
            response = requests.post(
                f"{api_url}/api/v1/query",
                json={"query": question, "top_k": 5},
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Query failed: {response.status_code}")
                continue
            
            data = response.json()
            
            result = {
                "question": question,
                "answer": data.get("answer", ""),
                "expected": expected,
                "sources_count": len(data.get("sources", [])),
                "processing_time": data.get("processing_time", 0)
            }
            
            results.append(result)
            
            logger.info(f"✓ Answer generated in {result['processing_time']:.2f}s")
        
        except Exception as e:
            logger.error(f"Error evaluating question: {e}")
    
    # Save results
    output_file = Path(questions_file).parent / "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nEvaluation complete!")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Questions evaluated: {len(results)}/{len(questions)}")
    
    # Print summary
    if results:
        avg_time = sum(r["processing_time"] for r in results) / len(results)
        logger.info(f"Average processing time: {avg_time:.2f}s")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate the RAG system")
    parser.add_argument(
        "questions_file",
        help="JSON file with test questions and expected answers"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the API"
    )
    
    args = parser.parse_args()
    
    try:
        evaluate_rag(args.questions_file, args.api_url)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
RAG Evaluation Test Script

Tests the Multimodal RAG system and generates comprehensive metrics.
"""

import sys
import logging
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_core import (
    process_pdf_and_create_vector_store,
    multimodal_pipeline_pdf_rag_pipeline,
    embed_text,
    save_vector_store,
    load_vector_store
)
from evaluation import RAGEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Test queries for evaluation
TEST_QUERIES = [
    {
        "query": "What is the main topic of this document?",
        "relevant_pages": None,  # Will be set based on actual document
        "category": "general"
    },
    {
        "query": "Summarize the key findings",
        "relevant_pages": None,
        "category": "summarization"
    },
    {
        "query": "What does the chart or diagram show?",
        "relevant_pages": None,
        "category": "visual"
    },
    {
        "query": "What are the main conclusions?",
        "relevant_pages": None,
        "category": "specific"
    },
    {
        "query": "Explain the methodology used",
        "relevant_pages": None,
        "category": "specific"
    }
]


def test_rag_system(pdf_path: str, output_path: str = "evaluation_results.json"):
    """
    Test RAG system and generate evaluation metrics.
    
    Args:
        pdf_path: Path to PDF file to test with
        output_path: Path to save results
    """
    logger.info("="*80)
    logger.info("MULTIMODAL RAG EVALUATION")
    logger.info("="*80)
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Load or process PDF
    logger.info(f"\n📄 Processing PDF: {pdf_path}")
    
    try:
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        
        logger.info("Processing document...")
        rag_data = process_pdf_and_create_vector_store(pdf_bytes)
        
        if not rag_data:
            logger.error("Failed to process PDF")
            return
        
        vector_store = rag_data['vector_store']
        image_data_store = rag_data['image_data_store']
        all_docs = rag_data['all_docs']
        
        logger.info(f"✅ Document processed successfully")
        logger.info(f"   - Total chunks: {len(all_docs)}")
        logger.info(f"   - Text chunks: {sum(1 for d in all_docs if d.metadata.get('type') == 'text')}")
        logger.info(f"   - Image chunks: {sum(1 for d in all_docs if d.metadata.get('type') == 'image')}")
        logger.info(f"   - Images stored: {len(image_data_store)}")
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return
    
    # Run test queries
    logger.info("\n" + "="*80)
    logger.info("RUNNING TEST QUERIES")
    logger.info("="*80)
    
    results = []
    
    for i, test_case in enumerate(TEST_QUERIES, 1):
        query = test_case['query']
        category = test_case['category']
        
        logger.info(f"\n📝 Query {i}/{len(TEST_QUERIES)} [{category}]")
        logger.info(f"   Q: {query}")
        
        try:
            # Run query with timing
            import time
            start_time = time.time()
            
            result = multimodal_pipeline_pdf_rag_pipeline(
                query=query,
                vector_store=vector_store,
                image_data_store=image_data_store
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            answer = result['answer']
            retrieved_docs = result['retrieved_docs']
            
            logger.info(f"   A: {answer[:150]}...")
            logger.info(f"   Retrieved: {len(retrieved_docs)} documents")
            
            # Evaluate
            metrics = evaluator.evaluate_rag_query(
                query=query,
                answer=answer,
                retrieved_docs=retrieved_docs,
                relevant_page_numbers=test_case.get('relevant_pages'),
                embed_func=embed_text,
                latency_ms=latency_ms
            )
            
            # Log metrics
            logger.info(f"\n   📊 Metrics:")
            for metric_name, value in metrics.items():
                logger.info(f"      {metric_name}: {value:.3f}")
            
            results.append({
                'query': query,
                'category': category,
                'answer': answer,
                'num_retrieved': len(retrieved_docs),
                'retrieved_pages': [doc.metadata.get('page') for doc in retrieved_docs[:5]],
                'metrics': metrics
            })
            
        except Exception as e:
            logger.error(f"   ❌ Error: {e}")
            results.append({
                'query': query,
                'category': category,
                'error': str(e)
            })
    
    # Aggregate results
    logger.info("\n" + "="*80)
    logger.info("AGGREGATED RESULTS")
    logger.info("="*80)
    
    successful_results = [r for r in results if 'metrics' in r]
    
    if successful_results:
        # Calculate averages
        avg_metrics = {}
        for metric_name in successful_results[0]['metrics'].keys():
            values = [r['metrics'][metric_name] for r in successful_results]
            avg_metrics[metric_name] = {
                'mean': float(sum(values) / len(values)),
                'min': float(min(values)),
                'max': float(max(values)),
                'std': float((sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5)
            }
        
        logger.info("\n📈 Average Metrics:")
        for metric_name, stats in avg_metrics.items():
            logger.info(f"   {metric_name}:")
            logger.info(f"      Mean: {stats['mean']:.3f}")
            logger.info(f"      Std:  {stats['std']:.3f}")
            logger.info(f"      Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        
        # Performance summary
        logger.info("\n⚡ Performance Summary:")
        latencies = [r['metrics']['latency_ms'] for r in successful_results if 'latency_ms' in r['metrics']]
        if latencies:
            logger.info(f"   Average Latency: {sum(latencies)/len(latencies):.2f}ms")
            logger.info(f"   Min Latency: {min(latencies):.2f}ms")
            logger.info(f"   Max Latency: {max(latencies):.2f}ms")
        
        # Quality assessment
        logger.info("\n🎯 Quality Assessment:")
        faithfulness_scores = [r['metrics']['faithfulness'] for r in successful_results]
        relevancy_scores = [r['metrics']['answer_relevancy'] for r in successful_results]
        
        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
        avg_relevancy = sum(relevancy_scores) / len(relevancy_scores)
        
        logger.info(f"   Faithfulness: {avg_faithfulness:.3f} {'✅' if avg_faithfulness > 0.7 else '⚠️'}")
        logger.info(f"   Relevancy: {avg_relevancy:.3f} {'✅' if avg_relevancy > 0.6 else '⚠️'}")
        
        if avg_faithfulness > 0.7 and avg_relevancy > 0.6:
            logger.info("\n   ✅ Model is performing well!")
        elif avg_faithfulness > 0.5 or avg_relevancy > 0.5:
            logger.info("\n   ⚠️ Model is performing adequately but has room for improvement")
        else:
            logger.info("\n   ❌ Model needs improvement")
    
    # Save results
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'pdf_path': pdf_path,
        'document_stats': {
            'total_chunks': len(all_docs),
            'text_chunks': sum(1 for d in all_docs if d.metadata.get('type') == 'text'),
            'image_chunks': sum(1 for d in all_docs if d.metadata.get('type') == 'image'),
        },
        'test_results': results,
        'aggregated_metrics': avg_metrics if successful_results else {}
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\n💾 Results saved to: {output_path}")
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Multimodal RAG system")
    parser.add_argument("pdf_path", help="Path to PDF file to test")
    parser.add_argument("--output", default="evaluation_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    test_rag_system(args.pdf_path, args.output)

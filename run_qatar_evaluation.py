"""
Qatar Document RAG Evaluation Script

Evaluates the Multimodal RAG system using the Qatar IMF report dataset.
Generates comprehensive metrics and a detailed markdown report.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_core import (
    process_pdf_and_create_vector_store,
    multimodal_pipeline_pdf_rag_pipeline,
    embed_text
)
from evaluation import RAGEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qatar_evaluation_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_evaluation_dataset(dataset_path: str) -> Dict[str, Any]:
    """Load the Qatar evaluation dataset from JSON file."""
    logger.info(f"Loading evaluation dataset from: {dataset_path}")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    logger.info(f"Loaded {dataset['total_questions']} questions")
    return dataset


def calculate_answer_similarity(generated: str, ground_truth: str, embed_func) -> float:
    """
    Calculate semantic similarity between generated and ground truth answers.
    
    Args:
        generated: Generated answer from RAG
        ground_truth: Ground truth answer
        embed_func: Embedding function
        
    Returns:
        Similarity score between 0 and 1
    """
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        
        gen_emb = embed_func(generated).reshape(1, -1)
        gt_emb = embed_func(ground_truth).reshape(1, -1)
        
        similarity = cosine_similarity(gen_emb, gt_emb)[0][0]
        # Normalize to 0-1 range
        return float((similarity + 1) / 2)
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return 0.0


def evaluate_qatar_dataset(
    pdf_path: str,
    dataset_path: str,
    output_md_path: str = "QATAR_EVALUATION_RESULTS.md"
):
    """
    Run comprehensive evaluation on Qatar dataset.
    
    Args:
        pdf_path: Path to Qatar PDF document
        dataset_path: Path to evaluation dataset JSON
        output_md_path: Path to save markdown results
    """
    logger.info("="*80)
    logger.info("QATAR DOCUMENT RAG EVALUATION")
    logger.info("="*80)
    
    # Load dataset
    dataset = load_evaluation_dataset(dataset_path)
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Process PDF
    logger.info(f"\n📄 Processing PDF: {pdf_path}")
    try:
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        
        logger.info("Processing document through RAG pipeline...")
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
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return
    
    # Run evaluation on all questions
    logger.info("\n" + "="*80)
    logger.info("RUNNING EVALUATION QUERIES")
    logger.info("="*80)
    
    results = []
    category_metrics = {}
    
    for i, question_data in enumerate(dataset['questions'], 1):
        question = question_data['question']
        category = question_data['category']
        ground_truth = question_data['ground_truth_answer']
        relevant_pages = question_data.get('relevant_pages', [])
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Question {i}/{dataset['total_questions']} [{category.upper()}]")
        logger.info(f"Q: {question}")
        logger.info(f"Ground Truth: {ground_truth[:100]}...")
        
        try:
            # Run query with timing
            import time
            start_time = time.time()
            
            result = multimodal_pipeline_pdf_rag_pipeline(
                query=question,
                vector_store=vector_store,
                image_data_store=image_data_store
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            answer = result['answer']
            retrieved_docs = result['retrieved_docs']
            
            logger.info(f"Generated Answer: {answer[:150]}...")
            logger.info(f"Retrieved: {len(retrieved_docs)} documents")
            
            # Evaluate retrieval and generation metrics
            metrics = evaluator.evaluate_rag_query(
                query=question,
                answer=answer,
                retrieved_docs=retrieved_docs,
                relevant_page_numbers=relevant_pages if relevant_pages else None,
                embed_func=embed_text,
                latency_ms=latency_ms
            )
            
            # Calculate answer correctness (semantic similarity with ground truth)
            answer_correctness = calculate_answer_similarity(
                answer, ground_truth, embed_text
            )
            metrics['answer_correctness'] = answer_correctness
            
            # Log metrics
            logger.info(f"\n📊 Metrics:")
            for metric_name, value in metrics.items():
                logger.info(f"   {metric_name}: {value:.3f}")
            
            # Store results
            result_entry = {
                'id': question_data['id'],
                'question': question,
                'category': category,
                'ground_truth': ground_truth,
                'generated_answer': answer,
                'num_retrieved': len(retrieved_docs),
                'retrieved_pages': [doc.metadata.get('page') for doc in retrieved_docs[:5]],
                'relevant_pages': relevant_pages,
                'metrics': metrics
            }
            results.append(result_entry)
            
            # Track category-wise metrics
            if category not in category_metrics:
                category_metrics[category] = []
            category_metrics[category].append(metrics)
            
        except Exception as e:
            logger.error(f"❌ Error processing question: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'id': question_data['id'],
                'question': question,
                'category': category,
                'error': str(e)
            })
    
    # Calculate aggregate metrics
    logger.info("\n" + "="*80)
    logger.info("CALCULATING AGGREGATE METRICS")
    logger.info("="*80)
    
    successful_results = [r for r in results if 'metrics' in r]
    
    if not successful_results:
        logger.error("No successful results to aggregate")
        return
    
    # Overall metrics
    all_metric_names = successful_results[0]['metrics'].keys()
    overall_metrics = {}
    
    for metric_name in all_metric_names:
        values = [r['metrics'][metric_name] for r in successful_results if metric_name in r['metrics']]
        if values:
            overall_metrics[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
    
    # Category-wise metrics
    category_aggregates = {}
    for category, metrics_list in category_metrics.items():
        category_aggregates[category] = {}
        for metric_name in all_metric_names:
            values = [m[metric_name] for m in metrics_list if metric_name in m]
            if values:
                category_aggregates[category][metric_name] = {
                    'mean': float(np.mean(values)),
                    'count': len(values)
                }
    
    # Generate markdown report
    logger.info(f"\n📝 Generating markdown report: {output_md_path}")
    generate_markdown_report(
        dataset=dataset,
        results=results,
        overall_metrics=overall_metrics,
        category_aggregates=category_aggregates,
        output_path=output_md_path,
        document_stats={
            'total_chunks': len(all_docs),
            'text_chunks': sum(1 for d in all_docs if d.metadata.get('type') == 'text'),
            'image_chunks': sum(1 for d in all_docs if d.metadata.get('type') == 'image')
        }
    )
    
    logger.info("\n" + "="*80)
    logger.info("✅ EVALUATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_md_path}")


def generate_markdown_report(
    dataset: Dict,
    results: List[Dict],
    overall_metrics: Dict,
    category_aggregates: Dict,
    output_path: str,
    document_stats: Dict
):
    """Generate comprehensive markdown evaluation report."""
    
    successful_results = [r for r in results if 'metrics' in r]
    failed_results = [r for r in results if 'error' in r]
    
    with open(output_path, 'w') as f:
        # Header
        f.write("# Qatar Document RAG Evaluation Results\n\n")
        f.write(f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Document:** {dataset['document_name']}\n\n")
        f.write(f"**Dataset:** {dataset['document_path']}\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Questions:** {dataset['total_questions']}\n")
        f.write(f"- **Successful Evaluations:** {len(successful_results)}\n")
        f.write(f"- **Failed Evaluations:** {len(failed_results)}\n")
        f.write(f"- **Success Rate:** {len(successful_results)/dataset['total_questions']*100:.1f}%\n\n")
        
        # Document Processing Stats
        f.write("### Document Processing\n\n")
        f.write(f"- **Total Chunks:** {document_stats['total_chunks']}\n")
        f.write(f"- **Text Chunks:** {document_stats['text_chunks']}\n")
        f.write(f"- **Image Chunks:** {document_stats['image_chunks']}\n\n")
        
        # Overall Performance Metrics
        f.write("## Overall Performance Metrics\n\n")
        f.write("| Metric | Mean | Std Dev | Min | Max |\n")
        f.write("|--------|------|---------|-----|-----|\n")
        
        for metric_name, stats in sorted(overall_metrics.items()):
            f.write(f"| {metric_name} | {stats['mean']:.3f} | {stats['std']:.3f} | "
                   f"{stats['min']:.3f} | {stats['max']:.3f} |\n")
        
        f.write("\n")
        
        # Performance Assessment
        f.write("### Performance Assessment\n\n")
        
        if 'answer_correctness' in overall_metrics:
            correctness = overall_metrics['answer_correctness']['mean']
            f.write(f"**Answer Correctness:** {correctness:.3f} ")
            if correctness >= 0.75:
                f.write("✅ Excellent\n")
            elif correctness >= 0.60:
                f.write("🟢 Good\n")
            elif correctness >= 0.45:
                f.write("🟡 Adequate\n")
            else:
                f.write("🔴 Needs Improvement\n")
        
        if 'faithfulness' in overall_metrics:
            faithfulness = overall_metrics['faithfulness']['mean']
            f.write(f"**Faithfulness:** {faithfulness:.3f} ")
            if faithfulness >= 0.70:
                f.write("✅ High\n")
            elif faithfulness >= 0.50:
                f.write("🟡 Moderate\n")
            else:
                f.write("🔴 Low\n")
        
        if 'answer_relevancy' in overall_metrics:
            relevancy = overall_metrics['answer_relevancy']['mean']
            f.write(f"**Answer Relevancy:** {relevancy:.3f} ")
            if relevancy >= 0.65:
                f.write("✅ High\n")
            elif relevancy >= 0.50:
                f.write("🟡 Moderate\n")
            else:
                f.write("🔴 Low\n")
        
        if 'latency_ms' in overall_metrics:
            latency = overall_metrics['latency_ms']['mean']
            f.write(f"**Average Latency:** {latency:.0f}ms ")
            if latency < 3000:
                f.write("✅ Fast\n")
            elif latency < 6000:
                f.write("🟡 Acceptable\n")
            else:
                f.write("🔴 Slow\n")
        
        f.write("\n")
        
        # Category-wise Performance
        f.write("## Performance by Question Category\n\n")
        f.write("| Category | Count | Answer Correctness | Faithfulness | Relevancy |\n")
        f.write("|----------|-------|-------------------|--------------|----------|\n")
        
        for category, metrics in sorted(category_aggregates.items()):
            count = metrics.get('answer_correctness', {}).get('count', 0)
            correctness = metrics.get('answer_correctness', {}).get('mean', 0)
            faithfulness = metrics.get('faithfulness', {}).get('mean', 0)
            relevancy = metrics.get('answer_relevancy', {}).get('mean', 0)
            
            f.write(f"| {category} | {count} | {correctness:.3f} | "
                   f"{faithfulness:.3f} | {relevancy:.3f} |\n")
        
        f.write("\n")
        
        # Detailed Results
        f.write("## Detailed Question-by-Question Results\n\n")
        
        for result in successful_results:
            f.write(f"### Question {result['id']}: {result['category'].upper()}\n\n")
            f.write(f"**Question:** {result['question']}\n\n")
            f.write(f"**Ground Truth Answer:**\n> {result['ground_truth']}\n\n")
            f.write(f"**Generated Answer:**\n> {result['generated_answer']}\n\n")
            
            # Metrics table
            f.write("**Metrics:**\n\n")
            f.write("| Metric | Score |\n")
            f.write("|--------|-------|\n")
            for metric_name, value in sorted(result['metrics'].items()):
                f.write(f"| {metric_name} | {value:.3f} |\n")
            
            # Retrieval info
            f.write(f"\n**Retrieved Pages:** {result['retrieved_pages'][:5]}\n")
            if result.get('relevant_pages'):
                f.write(f"**Relevant Pages (Ground Truth):** {result['relevant_pages']}\n")
            
            f.write("\n---\n\n")
        
        # Failed queries
        if failed_results:
            f.write("## Failed Queries\n\n")
            for result in failed_results:
                f.write(f"- **Question {result['id']}:** {result['question']}\n")
                f.write(f"  - **Error:** {result['error']}\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        if overall_metrics.get('answer_correctness', {}).get('mean', 0) < 0.60:
            f.write("- **Improve Answer Quality:** Consider fine-tuning the generation model or improving prompt engineering\n")
        
        if overall_metrics.get('faithfulness', {}).get('mean', 0) < 0.60:
            f.write("- **Enhance Faithfulness:** Strengthen grounding mechanisms to ensure answers stay closer to retrieved context\n")
        
        if 'precision@5' in overall_metrics and overall_metrics['precision@5']['mean'] < 0.60:
            f.write("- **Improve Retrieval:** Optimize embedding model or chunking strategy for better retrieval precision\n")
        
        if overall_metrics.get('latency_ms', {}).get('mean', 0) > 5000:
            f.write("- **Optimize Performance:** Consider caching, batch processing, or model optimization to reduce latency\n")
        
        f.write("\n---\n\n")
        f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG system on Qatar dataset")
    parser.add_argument("pdf_path", help="Path to Qatar PDF document")
    parser.add_argument("--dataset", default="qatar_evaluation_dataset.json",
                       help="Path to evaluation dataset JSON")
    parser.add_argument("--output", default="QATAR_EVALUATION_RESULTS.md",
                       help="Output markdown file path")
    
    args = parser.parse_args()
    
    evaluate_qatar_dataset(args.pdf_path, args.dataset, args.output)

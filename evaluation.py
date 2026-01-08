"""
RAG Evaluation Module

Implements comprehensive evaluation metrics for the Multimodal RAG system:
- Retrieval Metrics: Precision@K, Recall@K, MRR, NDCG
- Generation Metrics: Faithfulness, Answer Relevancy, Context Utilization
- Performance Metrics: Latency, Token Usage
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity
import time
import re

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluates RAG system performance across multiple dimensions."""
    
    def __init__(self, llm=None):
        """
        Initialize evaluator.
        
        Args:
            llm: Language model for LLM-as-judge evaluations
        """
        self.llm = llm
    
    # ==================== Retrieval Metrics ====================
    
    def precision_at_k(
        self, 
        retrieved_docs: List[Document], 
        relevant_page_numbers: List[int],
        k: int = 5
    ) -> float:
        """
        Calculate Precision@K - fraction of retrieved docs that are relevant.
        
        Args:
            retrieved_docs: Documents retrieved by the system
            relevant_page_numbers: List of page numbers that contain relevant info
            k: Number of top documents to consider
            
        Returns:
            Precision score between 0 and 1
        """
        if not retrieved_docs or not relevant_page_numbers:
            return 0.0
        
        top_k = retrieved_docs[:k]
        relevant_retrieved = sum(
            1 for doc in top_k 
            if doc.metadata.get('page') in relevant_page_numbers
        )
        
        precision = relevant_retrieved / k
        logger.info(f"Precision@{k}: {precision:.3f}")
        return precision
    
    def recall_at_k(
        self,
        retrieved_docs: List[Document],
        relevant_page_numbers: List[int],
        k: int = 5
    ) -> float:
        """
        Calculate Recall@K - fraction of relevant docs that were retrieved.
        
        Args:
            retrieved_docs: Documents retrieved by the system
            relevant_page_numbers: List of page numbers that contain relevant info
            k: Number of top documents to consider
            
        Returns:
            Recall score between 0 and 1
        """
        if not retrieved_docs or not relevant_page_numbers:
            return 0.0
        
        top_k = retrieved_docs[:k]
        relevant_retrieved = sum(
            1 for doc in top_k 
            if doc.metadata.get('page') in relevant_page_numbers
        )
        
        recall = relevant_retrieved / len(relevant_page_numbers)
        logger.info(f"Recall@{k}: {recall:.3f}")
        return recall
    
    def mean_reciprocal_rank(
        self,
        retrieved_docs: List[Document],
        relevant_page_numbers: List[int]
    ) -> float:
        """
        Calculate MRR - reciprocal rank of first relevant document.
        
        Args:
            retrieved_docs: Documents retrieved by the system
            relevant_page_numbers: List of page numbers that contain relevant info
            
        Returns:
            MRR score between 0 and 1
        """
        if not retrieved_docs or not relevant_page_numbers:
            return 0.0
        
        for rank, doc in enumerate(retrieved_docs, start=1):
            if doc.metadata.get('page') in relevant_page_numbers:
                mrr = 1.0 / rank
                logger.info(f"MRR: {mrr:.3f} (first relevant at rank {rank})")
                return mrr
        
        logger.info("MRR: 0.0 (no relevant docs retrieved)")
        return 0.0
    
    def ndcg_at_k(
        self,
        retrieved_docs: List[Document],
        relevant_page_numbers: List[int],
        k: int = 5
    ) -> float:
        """
        Calculate NDCG@K - normalized discounted cumulative gain.
        
        Args:
            retrieved_docs: Documents retrieved by the system
            relevant_page_numbers: List of page numbers that contain relevant info
            k: Number of top documents to consider
            
        Returns:
            NDCG score between 0 and 1
        """
        if not retrieved_docs or not relevant_page_numbers:
            return 0.0
        
        top_k = retrieved_docs[:k]
        
        # Calculate DCG
        dcg = 0.0
        for rank, doc in enumerate(top_k, start=1):
            relevance = 1 if doc.metadata.get('page') in relevant_page_numbers else 0
            dcg += relevance / np.log2(rank + 1)
        
        # Calculate IDCG (ideal DCG)
        ideal_relevances = [1] * min(len(relevant_page_numbers), k)
        idcg = sum(rel / np.log2(rank + 1) for rank, rel in enumerate(ideal_relevances, start=1))
        
        if idcg == 0:
            return 0.0
        
        ndcg = dcg / idcg
        logger.info(f"NDCG@{k}: {ndcg:.3f}")
        return ndcg
    
    # ==================== Generation Metrics ====================
    
    def faithfulness(
        self,
        answer: str,
        context_docs: List[Document]
    ) -> float:
        """
        Estimate faithfulness - whether answer is grounded in context.
        Uses simple heuristic: fraction of answer sentences that appear in context.
        
        Args:
            answer: Generated answer
            context_docs: Retrieved context documents
            
        Returns:
            Faithfulness score between 0 and 1
        """
        if not answer or not context_docs:
            return 0.0
        
        # Combine all context
        context_text = " ".join([
            doc.page_content for doc in context_docs 
            if doc.metadata.get('type') == 'text'
        ]).lower()
        
        # Split answer into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', answer) if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Check how many answer sentences have support in context
        supported = 0
        for sentence in sentences:
            # Extract key phrases (3+ words)
            words = sentence.lower().split()
            if len(words) < 3:
                continue
            
            # Check if significant overlap exists
            phrase_found = False
            for i in range(len(words) - 2):
                phrase = " ".join(words[i:i+3])
                if phrase in context_text:
                    phrase_found = True
                    break
            
            if phrase_found:
                supported += 1
        
        faithfulness_score = supported / len(sentences) if sentences else 0.0
        logger.info(f"Faithfulness: {faithfulness_score:.3f} ({supported}/{len(sentences)} sentences supported)")
        return faithfulness_score
    
    def answer_relevancy(
        self,
        query: str,
        answer: str,
        embed_func=None
    ) -> float:
        """
        Calculate answer relevancy using embedding similarity.
        
        Args:
            query: User query
            answer: Generated answer
            embed_func: Function to embed text
            
        Returns:
            Relevancy score between 0 and 1
        """
        if not query or not answer or not embed_func:
            return 0.0
        
        try:
            query_emb = embed_func(query).reshape(1, -1)
            answer_emb = embed_func(answer).reshape(1, -1)
            
            similarity = cosine_similarity(query_emb, answer_emb)[0][0]
            # Normalize to 0-1 range (cosine similarity is -1 to 1)
            relevancy = (similarity + 1) / 2
            
            logger.info(f"Answer Relevancy: {relevancy:.3f}")
            return float(relevancy)
        except Exception as e:
            logger.error(f"Error calculating answer relevancy: {e}")
            return 0.0
    
    def context_utilization(
        self,
        answer: str,
        context_docs: List[Document]
    ) -> float:
        """
        Estimate what fraction of context was used in the answer.
        
        Args:
            answer: Generated answer
            context_docs: Retrieved context documents
            
        Returns:
            Utilization score between 0 and 1
        """
        if not answer or not context_docs:
            return 0.0
        
        answer_lower = answer.lower()
        used_contexts = 0
        
        for doc in context_docs:
            if doc.metadata.get('type') != 'text':
                continue
            
            # Extract key phrases from context
            content = doc.page_content.lower()
            words = content.split()
            
            # Check if any significant phrase from context appears in answer
            context_used = False
            for i in range(len(words) - 2):
                phrase = " ".join(words[i:i+3])
                if len(phrase) > 10 and phrase in answer_lower:
                    context_used = True
                    break
            
            if context_used:
                used_contexts += 1
        
        text_docs = sum(1 for doc in context_docs if doc.metadata.get('type') == 'text')
        utilization = used_contexts / text_docs if text_docs > 0 else 0.0
        
        logger.info(f"Context Utilization: {utilization:.3f} ({used_contexts}/{text_docs} contexts used)")
        return utilization
    
    # ==================== Performance Metrics ====================
    
    def measure_latency(
        self,
        func,
        *args,
        **kwargs
    ) -> Tuple[Any, float]:
        """
        Measure function execution time.
        
        Args:
            func: Function to measure
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Tuple of (result, latency_ms)
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        latency_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Latency: {latency_ms:.2f}ms")
        return result, latency_ms
    
    # ==================== Comprehensive Evaluation ====================
    
    def evaluate_rag_query(
        self,
        query: str,
        answer: str,
        retrieved_docs: List[Document],
        relevant_page_numbers: Optional[List[int]] = None,
        embed_func=None,
        latency_ms: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Run comprehensive evaluation on a single query.
        
        Args:
            query: User query
            answer: Generated answer
            retrieved_docs: Documents retrieved by RAG
            relevant_page_numbers: Ground truth relevant pages (if available)
            embed_func: Embedding function for similarity calculations
            latency_ms: Query latency in milliseconds
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        # Retrieval metrics (only if ground truth available)
        if relevant_page_numbers:
            metrics['precision@5'] = self.precision_at_k(retrieved_docs, relevant_page_numbers, k=5)
            metrics['recall@5'] = self.recall_at_k(retrieved_docs, relevant_page_numbers, k=5)
            metrics['mrr'] = self.mean_reciprocal_rank(retrieved_docs, relevant_page_numbers)
            metrics['ndcg@5'] = self.ndcg_at_k(retrieved_docs, relevant_page_numbers, k=5)
        
        # Generation metrics
        metrics['faithfulness'] = self.faithfulness(answer, retrieved_docs)
        metrics['context_utilization'] = self.context_utilization(answer, retrieved_docs)
        
        if embed_func:
            metrics['answer_relevancy'] = self.answer_relevancy(query, answer, embed_func)
        
        # Performance metrics
        if latency_ms:
            metrics['latency_ms'] = latency_ms
        
        return metrics
    
    def evaluate_test_set(
        self,
        test_queries: List[Dict[str, Any]],
        rag_pipeline_func,
        embed_func=None
    ) -> Dict[str, Any]:
        """
        Evaluate RAG system on a test set of queries.
        
        Args:
            test_queries: List of dicts with 'query' and optionally 'relevant_pages'
            rag_pipeline_func: Function that takes query and returns answer + docs
            embed_func: Embedding function
            
        Returns:
            Dictionary with aggregated metrics
        """
        all_metrics = []
        
        for test_case in test_queries:
            query = test_case['query']
            relevant_pages = test_case.get('relevant_pages')
            
            # Run query with timing
            start_time = time.time()
            result = rag_pipeline_func(query)
            latency_ms = (time.time() - start_time) * 1000
            
            # Evaluate
            metrics = self.evaluate_rag_query(
                query=query,
                answer=result['answer'],
                retrieved_docs=result.get('retrieved_docs', []),
                relevant_page_numbers=relevant_pages,
                embed_func=embed_func,
                latency_ms=latency_ms
            )
            
            all_metrics.append(metrics)
        
        # Aggregate metrics
        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
        
        return {
            'individual_results': all_metrics,
            'aggregated_metrics': aggregated,
            'num_queries': len(test_queries)
        }

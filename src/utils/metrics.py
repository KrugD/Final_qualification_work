import logging
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


def compute_rouge(
    predictions: List[str],
    references: List[str],
    use_stemmer: bool = False,
) -> Dict[str, float]:
    """
    Compute ROUGE scores for summarization.
    
    Args:
        predictions: List of predicted summaries
        references: List of reference summaries
        use_stemmer: Whether to use stemming (False for Russian —
                     rouge_score uses English-only Porter stemmer)
    
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, ROUGE-L scores (F1)
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        logger.error("rouge_score not installed. Install with: pip install rouge-score")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=use_stemmer,
    )
    
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        # Handle empty strings
        if not pred.strip():
            pred = "empty"
        if not ref.strip():
            ref = "empty"
            
        result = scorer.score(ref, pred)
        
        for key in scores:
            scores[key].append(result[key].fmeasure)
        
        # Debug: log first 3 samples to understand ROUGE behavior
        if i < 3:
            logger.info(
                f"ROUGE debug sample {i}: "
                f"rouge1={result['rouge1'].fmeasure:.4f}, "
                f"rouge2={result['rouge2'].fmeasure:.4f}, "
                f"rougeL={result['rougeL'].fmeasure:.4f}, "
                f"pred_words={len(pred.split())}, ref_words={len(ref.split())}, "
                f"pred='{pred[:100]}', ref='{ref[:100]}'"
            )
    
    avg_scores = {
        "rouge1": float(np.mean(scores["rouge1"])),
        "rouge2": float(np.mean(scores["rouge2"])),
        "rougeL": float(np.mean(scores["rougeL"])),
    }
    
    # Log summary
    nonzero_r1 = sum(1 for s in scores["rouge1"] if s > 0)
    logger.info(
        f"ROUGE summary: {len(predictions)} samples, "
        f"{nonzero_r1} with rouge1>0, "
        f"avg rouge1={avg_scores['rouge1']:.6f}, "
        f"avg rougeL={avg_scores['rougeL']:.6f}"
    )
    
    return avg_scores


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    model_type: str = "DeepPavlov/rubert-base-cased",
    batch_size: int = 32,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute BERTScore for summarization.
    
    Args:
        predictions: List of predicted summaries
        references: List of reference summaries
        model_type: BERT model to use for scoring
        batch_size: Batch size for BERTScore computation
        device: Device to use (None for auto-detect)
    
    Returns:
        Dictionary with precision, recall, and F1 BERTScores
    """
    try:
        from bert_score import score
    except ImportError:
        logger.error("bert_score not installed. Install with: pip install bert-score")
        return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}
    
    # Handle empty predictions
    predictions = [p if p.strip() else "empty" for p in predictions]
    references = [r if r.strip() else "empty" for r in references]
    
    # Try primary model, fallback to multilingual
    models_to_try = [model_type, "bert-base-multilingual-cased"]
    
    for model in models_to_try:
        try:
            P, R, F1 = score(
                predictions,
                references,
                model_type=model,
                batch_size=batch_size,
                device=device,
                verbose=False,
            )
            
            if model != model_type:
                logger.info(f"BERTScore: using fallback model '{model}'")
            
            return {
                "bertscore_precision": P.mean().item(),
                "bertscore_recall": R.mean().item(),
                "bertscore_f1": F1.mean().item(),
            }
        except Exception as e:
            logger.warning(f"BERTScore with '{model}' failed: {e}")
            continue
    
    logger.error("BERTScore: all models failed")
    return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}


def compute_all_metrics(
    predictions: List[str],
    references: List[str],
    compute_bertscore_flag: bool = True,
    bertscore_model: str = "DeepPavlov/rubert-base-cased",
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute all summarization metrics.
    
    Args:
        predictions: List of predicted summaries
        references: List of reference summaries
        compute_bertscore_flag: Whether to compute BERTScore (slower)
        bertscore_model: Model for BERTScore
        device: Device for BERTScore computation
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # ROUGE scores
    rouge_scores = compute_rouge(predictions, references)
    metrics.update(rouge_scores)
    
    # BERTScore (optional, slower)
    if compute_bertscore_flag:
        bertscore_scores = compute_bertscore(
            predictions, references,
            model_type=bertscore_model,
            device=device,
        )
        metrics.update(bertscore_scores)
    
    return metrics


def compute_compression_ratio(
    predictions: List[str],
    sources: List[str],
) -> Dict[str, float]:
    """
    Compute compression ratio statistics.
    
    Args:
        predictions: List of predicted summaries
        sources: List of source texts
    
    Returns:
        Dictionary with compression ratio statistics
    """
    ratios = []
    
    for pred, src in zip(predictions, sources):
        src_len = len(src.split())
        pred_len = len(pred.split())
        
        if src_len > 0:
            ratios.append(pred_len / src_len)
    
    if not ratios:
        return {"compression_ratio_mean": 0.0, "compression_ratio_std": 0.0}
    
    return {
        "compression_ratio_mean": np.mean(ratios),
        "compression_ratio_std": np.std(ratios),
    }

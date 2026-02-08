import os
import logging
from typing import Dict, Optional, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


def setup_comet_logging(
    project_name: Optional[str] = None,
    experiment_name: Optional[str] = None,
    workspace: Optional[str] = None,
    tags: Optional[List[str]] = None,
    disabled: bool = False,
) -> Optional[Any]:
    """
    Setup CometML experiment for logging.
    
    Args:
        project_name: CometML project name (or from COMET_PROJECT_NAME env var)
        experiment_name: Name for this experiment
        workspace: CometML workspace (or from COMET_WORKSPACE env var)
        tags: Optional list of tags for the experiment
        disabled: If True, return None and don't create experiment
    
    Returns:
        CometML Experiment object or None if disabled/unavailable
    """
    if disabled:
        logger.info("CometML logging disabled")
        return None
    
    try:
        from comet_ml import Experiment
        
        api_key = os.getenv("COMET_API_KEY")
        if not api_key:
            logger.warning("COMET_API_KEY not found in environment, CometML disabled")
            return None
        
        project_name = project_name or os.getenv("COMET_PROJECT_NAME", "diffusion-summarization")
        workspace = workspace or os.getenv("COMET_WORKSPACE")
        
        experiment = Experiment(
            api_key=api_key,
            project_name=project_name,
            workspace=workspace,
            auto_metric_logging=True,
            auto_param_logging=True,
            log_code=True,
        )
        
        if experiment_name:
            experiment.set_name(experiment_name)
        
        if tags:
            experiment.add_tags(tags)
        
        logger.info(f"CometML experiment created: {experiment.get_key()}")
        return experiment
        
    except ImportError:
        logger.warning("comet_ml not installed, logging disabled")
        return None
    except Exception as e:
        logger.warning(f"Failed to create CometML experiment: {e}")
        return None


def log_metrics(
    experiment: Optional[Any],
    metrics: Dict[str, float],
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    prefix: str = "",
) -> None:
    """
    Log metrics to CometML.
    
    Args:
        experiment: CometML Experiment object
        metrics: Dictionary of metric names and values
        step: Global step number
        epoch: Epoch number
        prefix: Optional prefix for metric names (e.g., "train/", "val/")
    """
    if experiment is None:
        return
    
    try:
        for name, value in metrics.items():
            metric_name = f"{prefix}{name}" if prefix else name
            experiment.log_metric(metric_name, value, step=step, epoch=epoch)
    except Exception as e:
        logger.warning(f"Failed to log metrics: {e}")


def log_samples(
    experiment: Optional[Any],
    samples: List[Dict[str, str]],
    step: Optional[int] = None,
    table_name: str = "generation_samples",
) -> None:
    """
    Log generation samples to CometML.
    
    Args:
        experiment: CometML Experiment object
        samples: List of dicts with 'source', 'target', 'prediction' keys
        step: Global step number
        table_name: Name for the table in CometML
    """
    if experiment is None:
        return
    
    try:
        # Log as HTML table
        html_table = "<table border='1'><tr><th>Source</th><th>Target</th><th>Prediction</th></tr>"
        for sample in samples:
            source = sample.get("source", "")[:500]  # Truncate for display
            target = sample.get("target", "")[:200]
            prediction = sample.get("prediction", "")[:200]
            html_table += f"<tr><td>{source}</td><td>{target}</td><td>{prediction}</td></tr>"
        html_table += "</table>"
        
        experiment.log_html(html_table)
        
        # Also log as text
        for i, sample in enumerate(samples[:3]):  # Log first 3 as text
            experiment.log_text(
                f"Sample {i+1}:\nSource: {sample.get('source', '')[:300]}...\n"
                f"Target: {sample.get('target', '')}\n"
                f"Prediction: {sample.get('prediction', '')}",
                step=step,
            )
    except Exception as e:
        logger.warning(f"Failed to log samples: {e}")


def log_hyperparameters(
    experiment: Optional[Any],
    config: Dict[str, Any],
) -> None:
    """
    Log hyperparameters to CometML.
    
    Args:
        experiment: CometML Experiment object
        config: Dictionary of hyperparameters
    """
    if experiment is None:
        return
    
    try:
        experiment.log_parameters(config)
    except Exception as e:
        logger.warning(f"Failed to log hyperparameters: {e}")


def log_model(
    experiment: Optional[Any],
    model_path: str,
    model_name: str = "diffusion_summarizer",
) -> None:
    """
    Log model checkpoint to CometML.
    
    Args:
        experiment: CometML Experiment object
        model_path: Path to model checkpoint
        model_name: Name for the model in CometML
    """
    if experiment is None:
        return
    
    try:
        experiment.log_model(model_name, model_path)
    except Exception as e:
        logger.warning(f"Failed to log model: {e}")


def setup_logging(log_level: str = "INFO") -> None:
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

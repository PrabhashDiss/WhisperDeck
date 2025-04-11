"""Device and dtype selection utilities."""
import torch
import logging

logger = logging.getLogger(__name__)

def get_optimal_device():
    """
    Determine the optimal device (CUDA GPU, MPS, or CPU) for running the model.
    Returns:
        device: torch.device
        dtype: torch.dtype
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16  # Use half precision for CUDA
        logger.info("Using CUDA GPU")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32  # MPS requires full precision
        logger.info("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        logger.info("Using CPU")
    
    return device, dtype
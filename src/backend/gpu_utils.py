"""
GPU memory management utilities for SynCVE Backend.
Provides functions to manage GPU memory and prevent out-of-memory errors.
"""

import gc
from typing import Optional

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from deepface.commons.logger import Logger

logger = Logger()


def clear_gpu_memory(aggressive: bool = False) -> None:
    """
    Clear GPU memory with an optional aggressive mode.

    Args:
        aggressive (bool): When True, also clears TensorFlow sessions and resets
            GPU memory stats. Leave False for lightweight cleanup between requests.
    """
    try:
        # Run Python garbage collection
        gc.collect()

        if TENSORFLOW_AVAILABLE:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus and aggressive:
                # Clearing the TF session is expensive; only do it when explicitly requested
                if hasattr(tf.keras.backend, 'clear_session'):
                    tf.keras.backend.clear_session()

                for gpu in gpus:
                    try:
                        tf.config.experimental.reset_memory_stats(gpu)
                    except Exception as e:
                        logger.debug(f"Could not reset GPU memory stats: {e}")

        logger.debug("GPU memory cleared successfully")
    except Exception as e:
        logger.warn(f"Error clearing GPU memory: {e}")


def get_gpu_memory_usage() -> Optional[dict]:
    """
    Get current GPU memory usage statistics.

    Returns:
        dict: Dictionary with GPU memory info or None if GPU not available
    """
    try:
        if not TENSORFLOW_AVAILABLE:
            return None

        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return None

        memory_info = {}
        for i, gpu in enumerate(gpus):
            try:
                stats = tf.config.experimental.get_memory_info(gpu)
                memory_info[f'gpu_{i}'] = {
                    'current': stats.get('current', 0),
                    'peak': stats.get('peak', 0)
                }
            except Exception as e:
                logger.debug(f"Could not get GPU {i} memory info: {e}")

        return memory_info if memory_info else None
    except Exception as e:
        logger.debug(f"Error getting GPU memory usage: {e}")
        return None


def configure_gpu_memory(
    memory_fraction: Optional[float] = None,
    allow_growth: bool = True
) -> bool:
    """
    Configure GPU memory settings.

    Args:
        memory_fraction (float): Fraction of GPU memory to use (0.0-1.0)
        allow_growth (bool): Whether to allow dynamic memory growth

    Returns:
        bool: True if configuration was successful
    """
    try:
        if not TENSORFLOW_AVAILABLE:
            logger.warn("TensorFlow not available, cannot configure GPU")
            return False

        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            logger.warn("No GPU devices found")
            return False

        for gpu in gpus:
            try:
                if allow_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)

                if memory_fraction is not None and 0.0 < memory_fraction <= 1.0:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(
                            memory_limit=int(
                                tf.config.experimental.get_device_details(gpu)
                                .get('compute_capability', 1.0) * memory_fraction * 1024
                            )
                        )]
                    )
            except Exception as e:
                logger.warn(f"Error configuring GPU: {e}")
                continue

        logger.info("GPU memory configuration applied successfully")
        return True
    except Exception as e:
        logger.warn(f"Error in GPU memory configuration: {e}")
        return False


def limit_model_cache(max_models: int = 5) -> None:
    """
    Limit the number of cached models to prevent memory accumulation.

    Args:
        max_models (int): Maximum number of models to keep in cache
    """
    try:
        from deepface.modules import modeling

        if not hasattr(modeling, 'cached_models'):
            return

        for task in modeling.cached_models:
            models_dict = modeling.cached_models[task]
            if len(models_dict) > max_models:
                # Remove oldest models (FIFO)
                keys_to_remove = list(models_dict.keys())[:-max_models]
                for key in keys_to_remove:
                    try:
                        del models_dict[key]
                        logger.debug(f"Removed cached model: {task}/{key}")
                    except Exception as e:
                        logger.debug(f"Error removing cached model: {e}")
    except Exception as e:
        logger.warn(f"Error limiting model cache: {e}")

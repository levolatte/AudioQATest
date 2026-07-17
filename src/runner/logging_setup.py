"""Structured logging setup for the evaluation framework."""

import logging
import os
import sys
from pathlib import Path


def setup_logging(log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    """Configure the root logger with console and file handlers.

    Args:
        log_dir: Directory for log files.
        level: Log level string (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured root logger.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("audio_qa_eval")
    logger.setLevel(log_level)

    # Clear any existing handlers
    logger.handlers.clear()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(console)

    return logger


def get_task_logger(model: str, benchmark: str, perturbation: str,
                    log_dir: str = "logs") -> logging.Logger:
    """Create a per-task logger with a dedicated file handler.

    Args:
        model: Model identifier string.
        benchmark: Benchmark identifier string.
        perturbation: Perturbation identifier string.
        log_dir: Base log directory.

    Returns:
        Logger instance with file handler attached.
    """
    os.makedirs(log_dir, exist_ok=True)

    task_name = f"{model}__{benchmark}__{perturbation}"
    log_path = os.path.join(log_dir, f"{task_name}.log")

    logger = logging.getLogger(f"audio_qa_eval.{task_name}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    return logger

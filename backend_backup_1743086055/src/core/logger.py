import logging
import sys
from pathlib import Path
from loguru import logger as loguru_logger

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

def setup_logging(log_level: str = "INFO", json_logs: bool = False):
    # Remove default handlers
    loguru_logger.remove()

    # Create logs directory if it doesn't exist
    log_path = Path("logs")
    log_path.mkdir(exist_ok=True)

    # Configure loguru
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Add console handler
    loguru_logger.add(
        sys.stderr,
        format=log_format,
        level=log_level,
        colorize=True,
    )

    # Add file handlers
    loguru_logger.add(
        "logs/info.log",
        rotation="10 MB",
        retention="1 week",
        level="INFO",
        format=log_format,
        compression="zip",
    )

    loguru_logger.add(
        "logs/error.log",
        rotation="10 MB",
        retention="1 week",
        level="ERROR",
        format=log_format,
        compression="zip",
    )

    # Intercept standard library logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Replace logging handlers for third-party libraries
    for name in logging.root.manager.loggerDict:
        if name.startswith("uvicorn"):
            logging.getLogger(name).handlers = [InterceptHandler()]

    return loguru_logger

# Initialize logger
logger = setup_logging() 
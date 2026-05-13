"""Loguru configuration — JSON in prod, human-readable in dev; call configure_logging() once."""

import sys

from loguru import logger

from app.core.config import settings


def configure_logging() -> None:
    logger.remove()
    if settings.env == "prod":
        logger.add(sys.stdout, serialize=True, level=settings.log_level)
    else:
        logger.add(
            sys.stdout,
            colorize=True,
            level=settings.log_level,
            format=(
                "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level>"
                " | {message} | {extra}"
            ),
        )


__all__ = ["configure_logging", "logger"]

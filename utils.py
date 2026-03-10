# -*- coding: utf-8 -*-
"""Shared utilities: input validation, HTML sanitization, retry decorator, helpers."""

import re
import logging
import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type

import pandas as pd

logger = logging.getLogger(__name__)

# Valid subreddit names: 3–21 alphanumeric characters or underscores
_SUBREDDIT_RE = re.compile(r'^[A-Za-z0-9_]{3,21}$')


def validate_subreddit(name: str) -> str:
    """Validate and normalize a subreddit name.

    Strips leading 'r/' or '/' prefixes and whitespace, then checks the
    name against Reddit's naming rules (3–21 alphanumeric/underscore chars).

    Raises:
        ValueError: If the name does not match the expected pattern.
    """
    if not name:
        raise ValueError("Subreddit name must not be empty.")
    name = name.strip().lstrip('r/').lstrip('/')
    if not _SUBREDDIT_RE.match(name):
        raise ValueError(
            f"Invalid subreddit name: {name!r}. "
            "Must be 3–21 characters: letters, numbers, or underscores."
        )
    return name


def sanitize_markdown(text: str) -> str:
    """Escape HTML tags in text to prevent XSS when rendering LLM output.

    Only '<' and '>' are escaped so that markdown formatting (bold, bullets,
    etc.) is preserved while blocking injected HTML/script tags.
    """
    if not text:
        return text
    return text.replace('<', '&lt;').replace('>', '&gt;')


def retry(
    max_attempts: int = 3,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """Decorator: retry a function up to *max_attempts* times with exponential backoff.

    Args:
        max_attempts: Maximum number of call attempts (including the first).
        backoff: Initial wait in seconds; doubles on each subsequent retry.
        exceptions: Exception types that trigger a retry.
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = backoff
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        raise
                    logger.warning(
                        "%s attempt %d/%d failed: %s. Retrying in %.1fs.",
                        fn.__name__, attempt, max_attempts, exc, delay,
                    )
                    time.sleep(delay)
                    delay *= 2
        return wrapper
    return decorator


def extract_subreddit_name(url_or_name: str) -> Optional[str]:
    """Extract a subreddit name from a full Reddit URL or a plain name string."""
    if not url_or_name:
        return None
    try:
        url_match = re.search(r"(?:reddit\.com/r/)([^/]+)", url_or_name, re.IGNORECASE)
        if url_match:
            return url_match.group(1)
        # Treat as plain name if it has no slashes, spaces, or domain-like dots
        if "/" not in url_or_name and " " not in url_or_name and "." not in url_or_name:
            if not re.match(r".+\.[a-zA-Z]{2,4}$", url_or_name):
                return url_or_name
    except Exception as exc:
        logger.error("Error extracting subreddit name from %r: %s", url_or_name, exc)
    return None


def format_datetime(dt_object: Any) -> str:
    """Format a datetime-like value to a readable 'YYYY-MM-DD HH:MM:SS' string."""
    try:
        if isinstance(dt_object, datetime) and pd.notnull(dt_object):
            return dt_object.strftime('%Y-%m-%d %H:%M:%S')
        if pd.notnull(dt_object):
            return pd.to_datetime(dt_object).strftime('%Y-%m-%d %H:%M:%S')
        return ""
    except Exception as exc:
        logger.warning("Could not format datetime %r: %s", dt_object, exc)
        return str(dt_object)

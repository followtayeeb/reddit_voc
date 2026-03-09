# -*- coding: utf-8 -*-
"""Application-wide constants and environment configuration."""

import os
from dotenv import load_dotenv

try:
    load_dotenv()
except Exception:
    pass

APP_NAME = "RedditVOCAnalyzer"
APP_VERSION = "2.4"

REQUESTS_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/115.0.0.0 Safari/537.36"
)

# Timing
COMMENT_FETCH_DELAY = 0.3   # kept for reference; concurrent fetcher no longer sleeps
REQUEST_TIMEOUT = 20        # seconds for HTTP requests

# Cache
CACHE_TTL_SECONDS = 3600    # 1 hour

# Groq
GROQ_TEXT_MODELS = [
    "qwen/qwen3-32b",
    "gemma2-9b-it",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    # Preview models
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "qwen-qwq-32b",
    "mistral-saba-24b",
    "qwen-2.5-coder-32b",
    "qwen-2.5-32b",
    "deepseek-r1-distill-qwen-32b",
    "deepseek-r1-distill-llama-70b",
    "llama-3.3-70b-specdec",
    "llama-3.2-1b-preview",
    "llama-3.2-3b-preview",
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview",
]
DEFAULT_GROQ_MODEL = "llama3-8b-8192"

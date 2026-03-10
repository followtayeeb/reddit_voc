# -*- coding: utf-8 -*-
"""Shared fixtures for the reddit_voc test suite."""

import time
from unittest.mock import MagicMock

import pandas as pd
import pytest


@pytest.fixture
def sample_posts_df():
    """A minimal DataFrame mimicking fetched Reddit posts."""
    return pd.DataFrame([
        {
            'Post ID': 'abc123',
            'Title': 'Test post one',
            'Content': 'Some body text',
            'Score': 100,
            'Comments Count': 25,
            'Created Date': pd.Timestamp('2024-01-15 10:00:00'),
            'Author': 'user1',
            'URL': 'https://www.reddit.com/r/test/comments/abc123',
            'Sentiment Polarity': 0.1,
            'Sentiment Subjectivity': 0.4,
            'Sentiment Compound': 0.25,
        },
        {
            'Post ID': 'def456',
            'Title': 'Test post two',
            'Content': '',
            'Score': 50,
            'Comments Count': 10,
            'Created Date': pd.Timestamp('2024-01-14 08:00:00'),
            'Author': 'user2',
            'URL': 'https://www.reddit.com/r/test/comments/def456',
            'Sentiment Polarity': -0.2,
            'Sentiment Subjectivity': 0.5,
            'Sentiment Compound': -0.15,
        },
    ])


@pytest.fixture
def sample_comments_df():
    """A minimal DataFrame mimicking fetched Reddit comments."""
    return pd.DataFrame([
        {
            'Comment ID': 'c1',
            'Post ID': 'abc123',
            'Comment Body': 'This is a great product!',
            'Score': 42,
            'Created Date': pd.Timestamp('2024-01-15 11:00:00'),
            'Author': 'commenter1',
            'Is Submitter': False,
            'Sentiment Polarity': 0.5,
            'Sentiment Subjectivity': 0.6,
            'Sentiment Compound': 0.6249,
        },
        {
            'Comment ID': 'c2',
            'Post ID': 'abc123',
            'Comment Body': 'I had issues with delivery.',
            'Score': 10,
            'Created Date': pd.Timestamp('2024-01-15 12:00:00'),
            'Author': 'commenter2',
            'Is Submitter': False,
            'Sentiment Polarity': -0.3,
            'Sentiment Subjectivity': 0.4,
            'Sentiment Compound': -0.4215,
        },
    ])


@pytest.fixture
def mock_praw_reddit():
    """A MagicMock simulating a connected praw.Reddit instance."""
    mock = MagicMock()
    mock.auth.limits = {
        'remaining': 100,
        'used': 0,
        'reset_timestamp': time.time() + 600,
    }
    return mock


@pytest.fixture
def mock_groq_client():
    """A MagicMock simulating a connected groq.Groq client."""
    mock = MagicMock()
    choice = MagicMock()
    choice.message.content = "Test LLM response."
    mock.chat.completions.create.return_value = MagicMock(choices=[choice])
    mock.models.list.return_value = MagicMock(data=[])
    return mock

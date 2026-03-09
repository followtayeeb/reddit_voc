# -*- coding: utf-8 -*-
"""Reddit data-fetching functions: public JSON API and PRAW, plus concurrent helpers."""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import praw
import praw.models
import prawcore
import requests
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

from config import CACHE_TTL_SECONDS, REQUEST_TIMEOUT, REQUESTS_USER_AGENT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentiment helper (used inside cached fetch functions)
# ---------------------------------------------------------------------------

def get_sentiment(text: str) -> Dict[str, float]:
    """Compute TextBlob polarity/subjectivity and VADER compound for *text*.

    Falls back gracefully if either library fails or VADER hasn't been
    downloaded yet (tracked via st.session_state).
    """
    if not text or not isinstance(text, str):
        return {'polarity': 0.0, 'subjectivity': 0.0, 'compound': 0.0}

    polarity, subjectivity, compound = 0.0, 0.0, 0.0

    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
    except Exception as exc:
        logger.debug("TextBlob failed: %s", exc)

    try:
        if st.session_state.get('vader_downloaded', False):
            if 'sia' not in st.session_state:
                st.session_state.sia = SentimentIntensityAnalyzer()
            vader_scores = st.session_state.sia.polarity_scores(text)
            compound = vader_scores['compound']
    except Exception as exc:
        logger.debug("VADER failed: %s", exc)

    return {'polarity': polarity, 'subjectivity': subjectivity, 'compound': compound}


# ---------------------------------------------------------------------------
# Cached subreddit search
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Searching subreddits...")
def cached_search_subreddits(
    praw_details: Tuple[str, str, str],
    keyword: str,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Search Reddit subreddits by keyword via PRAW.

    Returns:
        (results_list, error_message) — one of them is None.
    """
    client_id, client_secret, user_agent = praw_details
    if not client_id or not client_secret:
        error_msg = "PRAW client_id or client_secret missing."
        logger.error("Subreddit search skipped for %r: %s", keyword, error_msg)
        return None, error_msg

    logger.info("CACHE MISS/EXPIRED: Searching subreddits for %r", keyword)
    try:
        temp_reddit = praw.Reddit(
            client_id=client_id, client_secret=client_secret,
            user_agent=user_agent, check_for_updates=False,
        )
        subreddits: List[Dict[str, Any]] = []
        for sub in temp_reddit.subreddits.search(keyword, limit=15):
            sub_name = getattr(sub, 'display_name', None)
            sub_title = getattr(sub, 'title', None)
            if sub_name and sub_title:
                subreddits.append({
                    'name': sub_name,
                    'title': sub_title,
                    'subscribers': getattr(sub, 'subscribers', 0) or 0,
                    'description': getattr(sub, 'public_description', 'N/A') or 'N/A',
                    'url': f"https://www.reddit.com/r/{sub_name}",
                })
        logger.info("Found %d subreddits for %r.", len(subreddits), keyword)
        return sorted(subreddits, key=lambda x: x['subscribers'], reverse=True), None

    except prawcore.exceptions.ResponseException as exc:
        status_code: Any = None
        if exc.response is not None:
            try:
                status_code = int(exc.response.status_code)
            except (ValueError, TypeError):
                status_code = 'Error'
        logger.error(
            "PRAW Response Error during search %r: Status %s – %s",
            keyword, status_code, exc, exc_info=True,
        )
        if status_code == 401:
            msg = "PRAW Authentication Failed (401). Check Client ID/Secret."
        elif status_code == 403:
            msg = "PRAW Forbidden (403). Check app permissions."
        elif status_code == 404:
            msg = "Reddit API endpoint not found (404)."
        elif isinstance(status_code, int) and status_code >= 500:
            msg = f"Reddit Server Error ({status_code}). Try again later."
        else:
            msg = f"Reddit API Response Error (Status: {status_code}) during search."
        return None, msg

    except prawcore.exceptions.RequestException as exc:
        logger.error("PRAW Network Error during search %r: %s", keyword, exc, exc_info=True)
        return None, "Network Error connecting to Reddit API. Check internet connection."

    except Exception as exc:
        logger.error("Unexpected error in subreddit search for %r: %s", keyword, exc, exc_info=True)
        return None, f"Unexpected error during subreddit search: {type(exc).__name__}"


# ---------------------------------------------------------------------------
# Cached post fetchers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Fetching public posts...")
def fetch_posts_requests(
    subreddit_name: str,
    limit: int,
    sort: str,
) -> Optional[pd.DataFrame]:
    """Fetch posts from r/{subreddit_name} via the public Reddit JSON API (no auth)."""
    logger.info(
        "CACHE MISS/EXPIRED: Public post fetch: r/%s, sort=%s, limit=%d",
        subreddit_name, sort, limit,
    )
    url = f"https://reddit.com/r/{subreddit_name}/{sort}.json?limit={limit}&t=all"
    headers = {'User-Agent': REQUESTS_USER_AGENT}
    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        if data.get('error') == 404:
            logger.warning("Requests: r/%s not found (404).", subreddit_name)
            return None
        if data.get('error') == 403:
            logger.warning("Requests: r/%s forbidden (private/quarantined?).", subreddit_name)
            return None
        if 'data' not in data or 'children' not in data['data']:
            logger.warning("Requests: Unexpected JSON structure from r/%s.", subreddit_name)
            return None

        posts_data = []
        count = 0
        for post in data['data']['children']:
            if count >= limit:
                break
            pdata = post.get('data', {})
            if post.get('kind') != 't3' or not pdata or pdata.get('stickied'):
                continue
            created_utc = pdata.get('created_utc')
            post_date = datetime.fromtimestamp(created_utc) if created_utc else datetime.now()
            sentiment = get_sentiment(f"{pdata.get('title', '')}. {pdata.get('selftext', '')}")
            posts_data.append({
                'Post ID': pdata.get('id', f'req_{int(time.time()*1000)}_{count}'),
                'Title': pdata.get('title', 'N/A'),
                'Content': pdata.get('selftext', ''),
                'Score': int(pdata.get('score', 0)),
                'Comments Count': int(pdata.get('num_comments', 0)),
                'Created Date': post_date,
                'Author': pdata.get('author', '[deleted]'),
                'URL': (
                    f"https://www.reddit.com{pdata['permalink']}"
                    if pdata.get('permalink') else pdata.get('url', '')
                ),
                'Sentiment Polarity': sentiment['polarity'],
                'Sentiment Subjectivity': sentiment['subjectivity'],
                'Sentiment Compound': sentiment['compound'],
            })
            count += 1

        if not posts_data:
            logger.info("Requests found no valid posts for r/%s.", subreddit_name)
            return None
        df = pd.DataFrame(posts_data)
        df['Created Date'] = pd.to_datetime(df['Created Date']).dt.tz_localize(None)
        logger.info("Requests fetched %d posts for r/%s.", len(df), subreddit_name)
        return df

    except requests.exceptions.Timeout:
        logger.error("Requests post fetch timed out for r/%s.", subreddit_name)
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 'N/A'
        logger.error("Requests HTTP %s for r/%s: %s", status, subreddit_name, exc)
    except requests.exceptions.RequestException as exc:
        logger.error("Requests post fetch failed for r/%s: %s", subreddit_name, exc, exc_info=True)
    except json.JSONDecodeError as exc:
        logger.error("JSON decode failed for r/%s posts: %s", subreddit_name, exc)
    except Exception as exc:
        logger.error("Unexpected error in public post fetch for r/%s: %s", subreddit_name, exc, exc_info=True)
    return None


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Fetching PRAW posts...")
def fetch_posts_praw(
    praw_details: Tuple[str, str, str],
    subreddit_name: str,
    sort_by: str,
    limit: int,
    start_date_ts: float,
    end_date_ts: float,
) -> Optional[pd.DataFrame]:
    """Fetch posts via PRAW with date-range filtering."""
    client_id, client_secret, user_agent = praw_details
    if not client_id or not client_secret:
        logger.error("PRAW fetch skipped: Missing credentials.")
        return None

    logger.info(
        "CACHE MISS/EXPIRED: PRAW post fetch: r/%s, sort=%s, limit=%d",
        subreddit_name, sort_by, limit,
    )
    start_date = datetime.fromtimestamp(start_date_ts)
    end_date = datetime.fromtimestamp(end_date_ts)

    try:
        temp_reddit = praw.Reddit(
            client_id=client_id, client_secret=client_secret,
            user_agent=user_agent, check_for_updates=False,
        )
        subreddit = temp_reddit.subreddit(subreddit_name)
        try:
            _ = subreddit.display_name
        except prawcore.exceptions.NotFound:
            logger.error("PRAW fetch: r/%s not found.", subreddit_name)
            return None
        except prawcore.exceptions.Redirect:
            logger.error("PRAW fetch: r/%s caused redirect (check spelling?).", subreddit_name)
            return None
        except prawcore.exceptions.Forbidden as exc:
            logger.error("PRAW fetch: r/%s is forbidden: %s", subreddit_name, exc)
            return None
        except Exception as exc:
            logger.error("PRAW fetch: error accessing r/%s: %s", subreddit_name, exc, exc_info=True)
            return None

        sort_map = {
            'Hot': subreddit.hot,
            'New': subreddit.new,
            'Top (Day)': lambda l: subreddit.top(time_filter='day', limit=l),
            'Top (Week)': lambda l: subreddit.top(time_filter='week', limit=l),
            'Top (Month)': lambda l: subreddit.top(time_filter='month', limit=l),
            'Top (Year)': lambda l: subreddit.top(time_filter='year', limit=l),
            'Top (All Time)': lambda l: subreddit.top(time_filter='all', limit=l),
            'Controversial (Day)': lambda l: subreddit.controversial(time_filter='day', limit=l),
            'Controversial (Week)': lambda l: subreddit.controversial(time_filter='week', limit=l),
            'Controversial (Month)': lambda l: subreddit.controversial(time_filter='month', limit=l),
            'Controversial (Year)': lambda l: subreddit.controversial(time_filter='year', limit=l),
            'Controversial (All Time)': lambda l: subreddit.controversial(time_filter='all', limit=l),
        }
        fetch_limit = limit + 50 if limit is not None else 75
        fetch_method = sort_map.get(sort_by, subreddit.hot)
        submissions = fetch_method(limit=fetch_limit)

        posts_data = []
        count = 0
        processed_ids: set = set()
        for post in submissions:
            if limit is not None and count >= limit:
                break
            if post.id in processed_ids or getattr(post, 'stickied', False):
                continue
            processed_ids.add(post.id)
            try:
                post_date = datetime.fromtimestamp(post.created_utc)
                if start_date <= post_date <= end_date:
                    sentiment = get_sentiment(
                        f"{getattr(post, 'title', '')}. {getattr(post, 'selftext', '')}"
                    )
                    posts_data.append({
                        'Post ID': post.id,
                        'Title': getattr(post, 'title', 'N/A'),
                        'Content': getattr(post, 'selftext', ''),
                        'Score': int(getattr(post, 'score', 0)),
                        'Comments Count': int(getattr(post, 'num_comments', 0)),
                        'Created Date': post_date,
                        'Author': str(post.author) if getattr(post, 'author', None) else '[deleted]',
                        'URL': (
                            f"https://www.reddit.com{post.permalink}"
                            if getattr(post, 'permalink', None) else getattr(post, 'url', '')
                        ),
                        'Sentiment Polarity': sentiment['polarity'],
                        'Sentiment Subjectivity': sentiment['subjectivity'],
                        'Sentiment Compound': sentiment['compound'],
                    })
                    count += 1
            except Exception as exc:
                logger.warning("Skipping post %s: %s", getattr(post, 'id', 'UNKNOWN'), exc)

        if not posts_data:
            logger.info("PRAW found no posts matching criteria for r/%s.", subreddit_name)
            return None
        df = pd.DataFrame(posts_data)
        df['Created Date'] = pd.to_datetime(df['Created Date']).dt.tz_localize(None)
        logger.info("PRAW fetched %d posts for r/%s.", len(df), subreddit_name)
        return df.sort_values(by='Created Date', ascending=False)

    except prawcore.exceptions.ResponseException as exc:
        status = exc.response.status_code if exc.response is not None else 'N/A'
        logger.error("PRAW Response error in post fetch r/%s: Status %s", subreddit_name, status, exc_info=True)
    except prawcore.exceptions.PrawcoreException as exc:
        logger.error("PRAW Core Error in post fetch r/%s: %s", subreddit_name, exc, exc_info=True)
    except Exception as exc:
        logger.error("Unexpected error in PRAW post fetch r/%s: %s", subreddit_name, exc, exc_info=True)
    return None


# ---------------------------------------------------------------------------
# Cached comment fetchers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Fetching public comments...")
def fetch_comments_requests(
    subreddit_name: str,
    post_id: str,
    limit: int,
) -> Optional[pd.DataFrame]:
    """Fetch comments for *post_id* via the public Reddit JSON API (no auth)."""
    logger.info(
        "CACHE MISS/EXPIRED: Public comment fetch: post %s, r/%s",
        post_id, subreddit_name,
    )
    fetch_limit = limit * 2 if limit < 100 else limit + 100
    url = (
        f"https://reddit.com/r/{subreddit_name}/comments/{post_id}"
        f".json?limit={fetch_limit}&depth=1&sort=top"
    )
    headers = {'User-Agent': REQUESTS_USER_AGENT}
    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        if (
            not isinstance(data, list) or len(data) < 2
            or not isinstance(data[1], dict)
            or 'data' not in data[1]
            or 'children' not in data[1]['data']
        ):
            logger.warning("Requests Comments: Bad JSON for post %s.", post_id)
            return None

        comments_data = []
        count = 0
        for comment in data[1]['data']['children']:
            if count >= limit:
                break
            cdata = comment.get('data', {})
            if (
                comment.get('kind') != 't1'
                or not cdata
                or cdata.get('body') in ['[deleted]', '[removed]', None, '']
                or not cdata.get('author')
            ):
                continue
            created_utc = cdata.get('created_utc')
            comment_date = datetime.fromtimestamp(created_utc) if created_utc else datetime.now()
            sentiment = get_sentiment(cdata.get('body', ''))
            comments_data.append({
                'Comment ID': cdata.get('id', f'req_comm_{int(time.time()*1000)}_{count}'),
                'Post ID': post_id,
                'Comment Body': cdata.get('body', ''),
                'Score': int(cdata.get('score', 0)),
                'Created Date': comment_date,
                'Author': cdata.get('author', '[unknown]'),
                'Is Submitter': cdata.get('is_submitter', False),
                'Sentiment Polarity': sentiment['polarity'],
                'Sentiment Subjectivity': sentiment['subjectivity'],
                'Sentiment Compound': sentiment['compound'],
            })
            count += 1

        if not comments_data:
            logger.info("Requests found no valid comments for post %s.", post_id)
            return None
        df = pd.DataFrame(comments_data)
        df['Created Date'] = pd.to_datetime(df['Created Date']).dt.tz_localize(None)
        logger.info("Requests fetched %d comments for post %s.", len(df), post_id)
        return df.sort_values(by='Score', ascending=False)

    except requests.exceptions.Timeout:
        logger.error("Requests comment fetch timed out for post %s.", post_id)
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 'N/A'
        logger.error("Requests HTTP %s for post %s comments: %s", status, post_id, exc)
    except requests.exceptions.RequestException as exc:
        logger.error("Requests comment fetch failed for post %s: %s", post_id, exc, exc_info=True)
    except json.JSONDecodeError as exc:
        logger.error("JSON decode failed for post %s comments: %s", post_id, exc)
    except Exception as exc:
        logger.error("Unexpected error in public comment fetch for post %s: %s", post_id, exc, exc_info=True)
    return None


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Fetching PRAW comments...")
def fetch_comments_praw(
    praw_details: Tuple[str, str, str],
    post_id: str,
    sort_by: str,
    limit: int,
) -> Optional[pd.DataFrame]:
    """Fetch comments for *post_id* via PRAW."""
    client_id, client_secret, user_agent = praw_details
    if not client_id or not client_secret:
        logger.error("PRAW comment fetch skipped: Missing credentials.")
        return None

    logger.info("CACHE MISS/EXPIRED: PRAW comment fetch: post %s, sort=%s", post_id, sort_by)
    comments_data = []
    try:
        temp_reddit = praw.Reddit(
            client_id=client_id, client_secret=client_secret,
            user_agent=user_agent, check_for_updates=False,
        )
        try:
            submission = temp_reddit.submission(id=post_id)
            _ = submission.title
        except (prawcore.exceptions.NotFound, prawcore.exceptions.Forbidden) as exc:
            logger.error("PRAW: Cannot access submission %s: %s", post_id, exc)
            return None
        except Exception as exc:
            logger.error("PRAW: Error accessing submission %s: %s", post_id, exc, exc_info=True)
            return None

        valid_sorts = ['confidence', 'top', 'new', 'controversial', 'old', 'random', 'qa', 'live']
        sort_by_lower = sort_by.lower()
        submission.comment_sort = sort_by_lower if sort_by_lower in valid_sorts else 'top'

        try:
            submission.comments.replace_more(limit=10, threshold=5)
        except prawcore.exceptions.ResponseException as exc:
            if exc.response and exc.response.status_code >= 500:
                logger.warning("PRAW replace_more server error for %s: %s", post_id, exc)
            else:
                logger.error("PRAW replace_more failed for %s: %s", post_id, exc)
        except Exception as exc:
            logger.error("PRAW replace_more unexpected error for %s: %s", post_id, exc, exc_info=True)

        comment_count = 0
        processed_ids: set = set()
        for comment in submission.comments.list():
            if comment_count >= limit:
                break
            if not isinstance(comment, praw.models.Comment) or comment.id in processed_ids:
                continue
            author = getattr(comment, 'author', None)
            body = getattr(comment, 'body', None)
            if not author or not body or body in ['[deleted]', '[removed]']:
                continue
            processed_ids.add(comment.id)
            try:
                sentiment = get_sentiment(body)
                comments_data.append({
                    'Comment ID': comment.id,
                    'Post ID': post_id,
                    'Comment Body': body,
                    'Score': int(getattr(comment, 'score', 0)),
                    'Created Date': datetime.fromtimestamp(comment.created_utc),
                    'Author': str(author),
                    'Is Submitter': getattr(comment, 'is_submitter', False),
                    'Sentiment Polarity': sentiment['polarity'],
                    'Sentiment Subjectivity': sentiment['subjectivity'],
                    'Sentiment Compound': sentiment['compound'],
                })
                comment_count += 1
            except Exception as exc:
                logger.warning("Skipping comment %s for post %s: %s",
                               getattr(comment, 'id', 'UNKNOWN'), post_id, exc)

        if not comments_data:
            logger.info("PRAW found no valid comments for post %s.", post_id)
            return None
        df = pd.DataFrame(comments_data)
        df['Created Date'] = pd.to_datetime(df['Created Date']).dt.tz_localize(None)
        logger.info("PRAW fetched %d comments for post %s.", len(df), post_id)
        return df

    except prawcore.exceptions.ResponseException as exc:
        status = exc.response.status_code if exc.response is not None else 'N/A'
        logger.error("PRAW Response error in comment fetch for %s: Status %s", post_id, status, exc_info=True)
    except prawcore.exceptions.PrawcoreException as exc:
        logger.error("PRAW Core Error in comment fetch for %s: %s", post_id, exc, exc_info=True)
    except Exception as exc:
        logger.error("Unexpected error in PRAW comment fetch for %s: %s", post_id, exc, exc_info=True)
    return None


# ---------------------------------------------------------------------------
# Concurrent comment fetching
# ---------------------------------------------------------------------------

def fetch_comments_for_post(
    subreddit_name: str,
    post_id: str,
    sort_by: str,
    limit: int,
    praw_details: Optional[Tuple[str, str, str]],
) -> Optional[pd.DataFrame]:
    """Fetch comments for a single post: PRAW first, public API as fallback."""
    if praw_details:
        try:
            praw_df = fetch_comments_praw(praw_details, post_id, sort_by, limit)
            if praw_df is not None and not praw_df.empty:
                return praw_df
        except Exception as exc:
            logger.error("PRAW comment fetch failed for %s: %s", post_id, exc, exc_info=True)

    try:
        df_req = fetch_comments_requests(subreddit_name, post_id, limit)
        if df_req is not None and not df_req.empty:
            return df_req
    except Exception as exc:
        logger.error("Public comment fetch failed for %s: %s", post_id, exc, exc_info=True)

    return None


def fetch_all_comments_concurrent(
    post_ids: List[str],
    subreddit_name: str,
    sort_by: str,
    limit: int,
    praw_details: Optional[Tuple[str, str, str]],
    max_workers: int = 5,
) -> Dict[str, Optional[pd.DataFrame]]:
    """Fetch comments for multiple posts concurrently using a thread pool.

    Returns a dict mapping post_id → DataFrame (or None on failure).
    All failures are caught and logged; the call never raises.
    """
    results: Dict[str, Optional[pd.DataFrame]] = {}
    logger.info(
        "Starting concurrent comment fetch for %d posts (max_workers=%d).",
        len(post_ids), max_workers,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_post = {
            executor.submit(
                fetch_comments_for_post,
                subreddit_name, post_id, sort_by, limit, praw_details,
            ): post_id
            for post_id in post_ids
        }
        for future in as_completed(future_to_post):
            post_id = future_to_post[future]
            try:
                df = future.result()
                results[post_id] = df
                if df is not None and not df.empty:
                    logger.info("Concurrent fetch OK for %s: %d comments.", post_id, len(df))
                else:
                    logger.warning("Concurrent fetch returned no comments for %s.", post_id)
            except Exception as exc:
                logger.error("Concurrent fetch raised for %s: %s", post_id, exc, exc_info=True)
                results[post_id] = None
    return results

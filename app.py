# -*- coding: utf-8 -*-
# ↑ Ensures UTF-8 encoding for potential special characters

import io
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import URLError

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import plotly.express as px
import praw
import prawcore
import streamlit as st
from dotenv import load_dotenv
from wordcloud import WordCloud

# --- New module imports ---
from config import (
    APP_NAME,
    APP_VERSION,
    CACHE_TTL_SECONDS,
    DEFAULT_GROQ_MODEL,
    GROQ_TEXT_MODELS,
)
from fetcher import (
    cached_search_subreddits,
    fetch_all_comments_concurrent,
    fetch_posts_praw,
    fetch_posts_requests,
    get_sentiment,
)
from utils import (
    extract_subreddit_name,
    format_datetime,
    sanitize_markdown,
    validate_subreddit,
)

# --- Setup Logging ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
log_handler = logging.FileHandler("app.log", mode='a', encoding='utf-8')
log_handler.setFormatter(log_formatter)
logger = logging.getLogger()
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)
logger.info("--- Streamlit App Starting / Reloading ---")

# Try to import Groq
try:
    from groq import Groq, RateLimitError
    GROQ_AVAILABLE = True
    logger.info("Groq library loaded.")
except ImportError:
    GROQ_AVAILABLE = False
    class RateLimitError(Exception): pass
    class Groq: pass  # Dummy class
    logger.warning("Groq library not found. LLM features will be disabled.")

# --- Environment Setup ---
try:
    dotenv_loaded = load_dotenv()
    logger.info(f".env file loaded: {dotenv_loaded}")
except Exception as e:
    logger.error(f"Error loading .env file: {e}", exc_info=True)

# --- NLTK VADER & Stopwords Download ---
if 'vader_downloaded' not in st.session_state:
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
        st.session_state.vader_downloaded = True
        logger.info("VADER lexicon found.")
    except LookupError:
        logger.info("Attempting VADER download...")
        try:
            nltk.download('vader_lexicon', quiet=True)
            st.session_state.vader_downloaded = True
            logger.info("VADER downloaded successfully.")
        except (URLError, OSError, Exception) as e:
            logger.error(f"Failed VADER download: Type {type(e).__name__} - {e}", exc_info=True)
            st.session_state.vader_downloaded = False

if 'stopwords_downloaded' not in st.session_state:
    try:
        nltk.data.find('corpora/stopwords')
        st.session_state.stopwords_downloaded = True
        logger.info("Stopwords found.")
    except LookupError:
        logger.info("Attempting Stopwords download...")
        try:
            nltk.download('stopwords', quiet=True)
            st.session_state.stopwords_downloaded = True
            logger.info("Stopwords downloaded successfully.")
        except (URLError, OSError, Exception) as e:
            logger.error(f"Failed stopwords download: Type {type(e).__name__} - {e}", exc_info=True)
            st.session_state.stopwords_downloaded = False


# --- LLM Analysis (cached) ---
@st.cache_data(ttl=CACHE_TTL_SECONDS * 2, show_spinner="Generating LLM analysis...")
def cached_llm_analysis(groq_api_key: str, comments_json_str: str, analysis_type: str, model_id: str) -> Optional[str]:
    logger.info(f"CACHE CHECK/MISS/EXPIRED: LLM analysis type '{analysis_type}' using model '{model_id}'")
    if not GROQ_AVAILABLE: return "LLM Error: Groq library not installed."
    if not groq_api_key: return "LLM Error: Groq API Key missing."
    if not model_id: return "LLM Error: Groq model not selected."
    if not comments_json_str or comments_json_str == '[]': return "LLM Info: No comment data passed for analysis."
    try:
        temp_groq_client = Groq(api_key=groq_api_key)
        try:
            comments_df = pd.read_json(io.StringIO(comments_json_str))
        except ValueError as json_err:
            logger.error(f"LLM Prep: Failed to parse comments JSON: {json_err}.")
            return "LLM Prep Error: Could not read comment data."
        if comments_df.empty: return "LLM Info: No comments data after parsing."
        max_comments_for_llm = 75
        sort_column = 'Score' if 'Score' in comments_df.columns else ('Created Date' if 'Created Date' in comments_df.columns else None)
        if sort_column:
            sampled_comments = comments_df.sort_values(by=sort_column, ascending=False).head(max_comments_for_llm)
        else:
            logger.warning("LLM Prep: Cannot sort comments, using first N.")
            sampled_comments = comments_df.head(max_comments_for_llm)
        comments_text_list = []
        for _, row in sampled_comments.iterrows():
            body = row.get('Comment Body', '')
            if body and isinstance(body, str):
                score_info = ""
                if sort_column and sort_column in row and pd.notna(row[sort_column]):
                    score_info = f" (Score: {row[sort_column]})"
                truncated_body = body[:400] + ('...' if len(body) > 400 else '')
                comments_text_list.append(f"- {truncated_body}{score_info}")
        comments_text = "\n".join(comments_text_list)
        if not comments_text.strip(): return "LLM Info: No valid comment text prepared after filtering/sampling."
        system_prompt = "You are an AI assistant specialized in analyzing Reddit comment sections.\nYou will receive a sample of comments (potentially truncated and sorted by score) from a specific Reddit post.\nYour goal is to provide a concise, insightful analysis focused *strictly* on the requested aspect. Use clear bullet points for lists."
        user_prompt_header = f"Based *only* on the following sample of comments, please provide an analysis focusing on **{analysis_type}**.\n\n--- START OF COMMENT DATA ---\n{comments_text}\n--- END OF COMMENT DATA ---\n\n**Analysis of '{analysis_type}':**"
        if analysis_type == "Overall Summary":
            instructions = "Provide a brief (3-4 bullet points) overview covering:\n- The main subjects being discussed.\n- The general sentiment or tone (e.g., positive, negative, mixed, specific emotions).\n- Any standout observations."
        elif analysis_type == "Themes and Topics":
            instructions = "Identify and list the 3-5 most prominent and distinct themes or topics discussed in these comments. Briefly describe each theme in one sentence."
        elif analysis_type == "Sentiment Analysis (Detailed)":
            instructions = "Analyze the sentiment expressed:\n- What is the dominant overall sentiment (positive, negative, neutral, mixed)?\n- What specific tones or emotions are prevalent (e.g., appreciation, frustration, humor, sarcasm, critique, confusion)? Provide brief examples indirectly (e.g., 'comments expressing frustration about X')."
        elif analysis_type == "Pain Points / Complaints":
            instructions = "Extract and summarize in bullet points the **main** pain points, problems, complaints, or negative feedback mentioned by the commenters regarding the subject of the post. Be specific."
        elif analysis_type == "Positive Feedback / Praise":
            instructions = "Extract and summarize in bullet points the **main** points of positive feedback, praise, agreement, or appreciation mentioned by the commenters regarding the subject of the post. Be specific."
        elif analysis_type == "Actionable Insights / Suggestions":
            instructions = "Identify and list in bullet points any **actionable insights, specific suggestions, or concrete recommendations** made by the commenters. Focus on ideas that could lead to improvements or changes, not just general opinions."
        else:
            instructions = f"Provide your analysis based on the requested focus: '{analysis_type}'."
        full_user_prompt = f"{user_prompt_header}\n\n{instructions}"
        logger.info(f"Calling Groq API model '{model_id}' for analysis type '{analysis_type}'.")
        start_time = time.time()
        chat_completion = temp_groq_client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": full_user_prompt}],
            model=model_id, temperature=0.5, max_tokens=1024, top_p=1, stop=None, stream=False,
        )
        duration = time.time() - start_time
        analysis = chat_completion.choices[0].message.content
        logger.info(f"Groq analysis successful for '{analysis_type}' in {duration:.2f}s.")
        return analysis
    except RateLimitError:
        logger.error("Groq API rate limit exceeded during analysis.")
        return "LLM Error: Groq API rate limit exceeded. Please wait and try again."
    except pd.errors.EmptyDataError:
        logger.error("LLM Prep: Pandas could not read empty JSON data.")
        return "LLM Info: Could not process empty comments data."
    except Exception as e:
        logger.error(f"Error in cached LLM analysis ({type(e).__name__}): {e}", exc_info=True)
        return f"LLM Error: An unexpected error occurred during analysis - {type(e).__name__}"


# --- Core Analyzer Class ---
class RedditVOCAnalyzer:
    def __init__(self):
        self.praw_details: Optional[Tuple[str, str, str]] = None
        self.groq_api_key: Optional[str] = None
        logger.info("RedditVOCAnalyzer instance created.")

    def connect_praw(self, client_id: str, client_secret: str, app_name: str, username: Optional[str] = None) -> Tuple[bool, str]:
        """Connects to PRAW using Script App credentials and constructs User-Agent."""
        logger.info(f"Attempting PRAW connection (ID: {client_id[:4]}..., App: {app_name}, User: {username or 'None'}).")
        if not client_id or not client_secret or not app_name:
            msg = "❌ PRAW Failed: Client ID, Client Secret, and App Name are required."
            logger.error(msg); return False, msg

        platform = "web"
        user_agent_base = f"{platform}:{app_name}:{APP_VERSION}"
        if username:
            clean_username = username.replace('/u/', '').strip()
            user_agent = f"{user_agent_base} (by /u/{clean_username})" if clean_username else user_agent_base
        else:
            user_agent = user_agent_base
        logger.info(f"Using PRAW User-Agent: {user_agent}")

        config = {"user_agent": user_agent, "check_for_updates": False, "client_id": client_id, "client_secret": client_secret}
        try:
            test_reddit = praw.Reddit(**config)
            limits = test_reddit.auth.limits
            remaining = limits.get('remaining', '?')
            reset_timestamp = limits.get('reset_timestamp')
            reset_mins = '?'
            if reset_timestamp is not None:
                try:
                    reset_secs = float(reset_timestamp) - time.time()
                    reset_mins = max(0, int(reset_secs / 60))
                except (ValueError, TypeError) as time_err:
                    logger.warning(f"Could not calculate rate limit reset time: {time_err}")
            msg = f"✅ PRAW Connected (Limit: {remaining} rem, ~{reset_mins} min reset)"
            self.praw_details = (client_id, client_secret, user_agent)
            logger.info("PRAW connection successful.")
            return True, msg
        except prawcore.exceptions.OAuthException as e:
            self.praw_details = None
            msg = f"❌ PRAW Auth Error (OAuth): Invalid credentials or permissions. Verify ID/Secret & app type ('script'). Details: {e}"
            logger.error(f"PRAW OAuth Error: {e}", exc_info=True)
            return False, msg
        except prawcore.exceptions.ResponseException as e:
            self.praw_details = None
            status = e.response.status_code if e.response else 'N/A'
            if status == 401:
                msg = "❌ PRAW Auth Error (401): Invalid Client ID or Secret. Please re-verify on Reddit."
            else:
                msg = f"❌ PRAW API Error ({status}): Problem connecting. Check status/wait. Details: {e}"
            logger.error(f"PRAW Response Error on connect: {e}", exc_info=True)
            return False, msg
        except prawcore.exceptions.RequestException as e:
            self.praw_details = None
            msg = f"❌ PRAW Network Error: Could not reach Reddit. Check connection. Details: {e}"
            logger.error(f"PRAW Network Error on connect: {e}", exc_info=True)
            return False, msg
        except Exception as e:
            self.praw_details = None
            msg = f"❌ PRAW Connection Failed (Unexpected): {e}"
            logger.exception("Unexpected PRAW connection failed:")
            return False, msg

    def connect_groq(self, api_key: str) -> Tuple[bool, str]:
        logger.info("Attempting Groq connection...")
        if not GROQ_AVAILABLE:
            msg = "Groq library not installed."
            logger.warning(msg)
            return False, msg
        if not api_key:
            msg = "❌ Groq Failed: API Key required."
            logger.error(msg)
            return False, msg
        try:
            temp_groq_client = Groq(api_key=api_key)
            _ = temp_groq_client.models.list()
            self.groq_api_key = api_key
            msg = "✅ Groq Connected & Key Validated"
            logger.info("Groq connection successful and key validated.")
            return True, msg
        except RateLimitError:
            self.groq_api_key = None
            msg = "❌ Groq Connection Failed: Rate limit hit during initial check."
            logger.error("Groq connection failed - Rate Limit Error", exc_info=True)
            return False, msg
        except Exception as e:
            self.groq_api_key = None
            err_detail = str(e)
            is_auth_error = (
                "invalid api key" in err_detail.lower()
                or "authentication" in err_detail.lower()
                or (hasattr(e, "status_code") and e.status_code == 401)
            )
            if is_auth_error:
                msg = f"❌ Groq Auth Failed: Invalid API Key. Please check your key. Details: {err_detail}"
            else:
                msg = f"❌ Groq Connection Failed ({type(e).__name__}): {err_detail}"
            logger.error(f"Groq connection failed: {type(e).__name__} - {err_detail}", exc_info=True)
            return False, msg

    def search_subreddits(self, keyword: str) -> List[Dict[str, Any]]:
        st.session_state._search_had_error = False
        if not self.praw_details:
            st.error("Cannot search: PRAW connection needed. Please connect in the sidebar.")
            logger.warning("Search skipped: PRAW not connected.")
            return []
        results, error_msg = cached_search_subreddits(self.praw_details, keyword)
        if error_msg:
            logger.error(f"Subreddit search for '{keyword}' failed: {error_msg}")
            st.error(f"Search Error: {error_msg}")
            st.session_state._search_had_error = True
            return []
        elif results is None:
            logger.error(f"Subreddit search for '{keyword}' unexpected None result.")
            st.error("Unexpected issue during search (None result). Check logs.")
            st.session_state._search_had_error = True
            return []
        elif not results:
            logger.info(f"Subreddit search for '{keyword}' returned no results.")
            return []
        else:
            logger.info(f"Subreddit search for '{keyword}' returned {len(results)} result(s).")
            return results

    def fetch_posts_multimethod(self, subreddit_name: str, sort_by: str, limit: int, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        final_df = None
        basic_sort = sort_by.lower().split(" ")[0]
        public_sort_map = {'hot': 'hot', 'new': 'new', 'top': 'top', 'controversial': 'controversial'}
        public_sort = public_sort_map.get(basic_sort, 'hot')
        logger.info(f"Initiating multi-method post fetch for r/{subreddit_name} (Sort: {sort_by}, Limit: {limit})")
        try:
            public_fetch_limit = limit + 30 if limit < 70 else 100
            df_req = fetch_posts_requests(subreddit_name, public_fetch_limit, public_sort)
            if df_req is not None and not df_req.empty:
                df_req['Created Date'] = pd.to_datetime(df_req['Created Date']).dt.tz_localize(None)
                start_date_naive = start_date.replace(tzinfo=None)
                end_date_naive = end_date.replace(tzinfo=None)
                filtered_df = df_req[(df_req['Created Date'] >= start_date_naive) & (df_req['Created Date'] <= end_date_naive)].copy()
                if not filtered_df.empty:
                    final_df = filtered_df.head(limit)
                    final_df['Created Date'] = final_df['Created Date'].dt.tz_localize(None)
                    logger.info(f"Post fetch SUCCESS (Public): Found {len(final_df)} posts for r/{subreddit_name}.")
                    return final_df.sort_values(by='Created Date', ascending=False)
                else:
                    logger.info(f"Public fetch yielded posts, but none matched date range.")
            else:
                logger.warning(f"Public fetch returned None or empty for r/{subreddit_name}.")
        except Exception as e:
            logger.error(f"Error during public post fetch for r/{subreddit_name}: {e}", exc_info=True)

        if self.praw_details:
            try:
                praw_df = fetch_posts_praw(self.praw_details, subreddit_name, sort_by, limit, start_date.timestamp(), end_date.timestamp())
                if praw_df is not None and not praw_df.empty:
                    logger.info(f"Post fetch SUCCESS (PRAW): Found {len(praw_df)} posts for r/{subreddit_name}.")
                    return praw_df
                else:
                    logger.warning(f"PRAW fallback found no posts for r/{subreddit_name}.")
            except Exception as e:
                logger.error(f"PRAW fallback fetch failed for r/{subreddit_name}: {e}", exc_info=True)
        else:
            logger.info("PRAW fallback skipped: PRAW not connected.")

        if final_df is not None and not final_df.empty:
            return final_df.sort_values(by='Created Date', ascending=False)
        logger.error(f"Multi-method post fetch FAILED entirely for r/{subreddit_name}.")
        return None

    def generate_llm_analysis(self, comments_df: pd.DataFrame, analysis_type: str, model_id: str) -> Optional[str]:
        logger.info(f"Initiating LLM analysis: '{analysis_type}' using model '{model_id}'")
        if not self.groq_api_key:
            msg = "LLM Error: Groq not connected or key missing."
            logger.error(msg)
            return msg
        if comments_df is None or comments_df.empty:
            msg = "LLM Info: No comments data provided to analyze."
            logger.warning(msg)
            return msg
        if not isinstance(comments_df, pd.DataFrame):
            msg = "LLM Prep Error: Invalid data format for comments."
            logger.error(msg)
            return msg
        if not model_id:
            msg = "LLM Error: No Groq model selected."
            logger.error(msg)
            return msg
        try:
            required_cols = ['Comment Body']
            optional_cols = ['Score', 'Created Date']
            cols_to_use = required_cols + [col for col in optional_cols if col in comments_df.columns]
            if 'Comment Body' not in cols_to_use:
                msg = "LLM Prep Error: 'Comment Body' column missing."
                logger.error(msg)
                return msg
            comments_to_analyze = comments_df[cols_to_use]
            comments_json_str = comments_to_analyze.to_json(orient="records", date_format="iso", default_handler=str)
            return cached_llm_analysis(self.groq_api_key, comments_json_str, analysis_type, model_id)
        except Exception as e:
            msg = f"LLM Prep Error: Failed to prepare comments - {e}"
            logger.error(msg, exc_info=True)
            return msg


# --- Visualization Functions ---
def plot_sentiment_distribution(df: pd.DataFrame, column: str = 'Sentiment Compound'):
    if df is None or df.empty or column not in df.columns:
        logger.warning(f"Skipping sentiment plot: No data or missing column '{column}'.")
        st.info("Not enough valid comment data for sentiment plot.")
        return
    plot_df = df.dropna(subset=[column])
    if plot_df.empty:
        logger.warning("Skipping sentiment plot: No valid scores after dropping NAs.")
        st.info("No valid sentiment scores found for plot.")
        return
    try:
        fig = px.histogram(
            plot_df, x=column, nbins=20,
            title=f'Distribution of Comment Sentiment ({column})',
            labels={column: 'Sentiment Score (-1 to +1)'},
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_layout(bargap=0.1, xaxis_title="Sentiment Score", yaxis_title="Number of Comments")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        logger.error(f"Failed sentiment plot: {e}", exc_info=True)
        st.error("Could not generate sentiment plot.")


def generate_word_cloud(text_series: pd.Series, title: str = "Word Cloud"):
    if text_series is None or text_series.empty:
        logger.warning("Skipping word cloud: No text data.")
        st.info("Not enough comment text for word cloud.")
        return
    try:
        valid_texts = text_series.dropna().astype(str)
        text = " ".join(comment for comment in valid_texts if comment.strip())
    except Exception as e:
        logger.error(f"Error processing text for word cloud: {e}", exc_info=True)
        st.error("Could not process text for word cloud.")
        return
    if not text.strip():
        logger.warning("Skipping word cloud: No valid text content.")
        st.info("No valid words found for word cloud.")
        return
    stopwords_list = None
    if st.session_state.get('stopwords_downloaded', False):
        try:
            stopwords_list = set(nltk.corpus.stopwords.words('english'))
        except LookupError:
            logger.warning("Stopwords lookup failed. Re-downloading.")
            st.session_state.stopwords_downloaded = False
        except Exception as e:
            logger.error(f"Error accessing stopwords: {e}", exc_info=True)
            st.error("Could not load stopwords.")
            return
    if not stopwords_list and not st.session_state.get('stopwords_downloaded', False):
        try:
            nltk.download('stopwords', quiet=True)
            stopwords_list = set(nltk.corpus.stopwords.words('english'))
            st.session_state.stopwords_downloaded = True
        except Exception as dl_err:
            logger.error(f"Stopwords download failed: {dl_err}")
            st.error("Word cloud needs stopwords; download failed.")
            return
    if not stopwords_list:
        logger.error("Word cloud skipped: Stopwords unavailable.")
        st.error("Word cloud requires stopwords, but they couldn't be loaded/downloaded.")
        return
    try:
        logger.info(f"Generating word cloud: '{title}'")
        wordcloud = WordCloud(
            width=800, height=400, background_color='white', colormap='viridis',
            max_words=150, contour_width=1, contour_color='steelblue',
            stopwords=stopwords_list, collocations=True, normalize_plurals=True,
            prefer_horizontal=0.9,
        ).generate(text)
        if not wordcloud.words_:
            logger.warning(f"Word cloud '{title}' has no words.")
            st.info("No significant words found for word cloud.")
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.subheader(title)
        st.pyplot(fig)
        img_buf = io.BytesIO()
        try:
            fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=150)
            plt.close(fig)
            img_buf.seek(0)
            st.download_button(
                label="💾 Download Word Cloud",
                data=img_buf,
                file_name=f"{title.lower().replace(' ', '_')}_wordcloud.png",
                mime="image/png",
            )
        except Exception as save_err:
            if 'fig' in locals() and plt.fignum_exists(fig.number):
                plt.close(fig)
            logger.error(f"Could not save word cloud '{title}': {save_err}", exc_info=True)
            st.error("Could not prepare word cloud for download.")
    except Exception as e:
        logger.error(f"Could not generate word cloud '{title}': {e}", exc_info=True)
        st.error(f"Could not generate word cloud: {e}")
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)


# --- Streamlit App UI ---
def main():
    st.set_page_config(page_title="Reddit VOC Analyzer", page_icon="📊", layout="wide")
    st.title("📊 Reddit Voice of Customer (VOC) Analyzer")
    st.write("Analyzes subreddit posts/comments using public methods first, with PRAW API as fallback/for full comments.")
    st.caption(f"Version {APP_VERSION}. Having trouble? Check the 📖 `Guide` page (if available) or the `app.log` file for errors.")

    # Initialize Session State
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = RedditVOCAnalyzer()
    st.session_state.setdefault('praw_connection_status', "PRAW Not Connected.")
    st.session_state.setdefault('groq_connection_status', "Groq Not Connected.")
    st.session_state.setdefault('posts_df', None)
    st.session_state.setdefault('comments_data', {})
    st.session_state.setdefault('selected_post_ids', [])
    st.session_state.setdefault('subreddit_target', None)
    st.session_state.setdefault('fetch_attempted', False)
    st.session_state.setdefault('selected_groq_model', DEFAULT_GROQ_MODEL)
    st.session_state.setdefault('_search_had_error', False)

    analyzer: RedditVOCAnalyzer = st.session_state.analyzer
    praw_connected = analyzer.praw_details is not None
    groq_connected = analyzer.groq_api_key is not None

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        with st.expander("PRAW Connection (Recommended)", expanded=not praw_connected):
            st.caption("Connect for reliable fetching, sorting & search.")
            cred_client_id = st.text_input("Reddit Client ID", value=os.getenv('REDDIT_CLIENT_ID', ''), type="password", key="cred_id", help="From reddit.com/prefs/apps.")
            cred_client_secret = st.text_input("Client Secret", value=os.getenv('REDDIT_CLIENT_SECRET', ''), type="password", key="cred_secret", help="From reddit.com/prefs/apps.")
            st.caption("For Reddit API User-Agent:")
            reddit_app_name = st.text_input("Reddit App Name", value=os.getenv('REDDIT_APP_NAME', APP_NAME), key="reddit_app_name", help="The name you gave your script app on reddit.com/prefs/apps.")
            reddit_username = st.text_input("Your Reddit Username (Optional)", value=os.getenv('REDDIT_USERNAME', ''), key="reddit_username", help="Optional, but recommended for API compliance.")
            can_connect_praw = cred_client_id and cred_client_secret and reddit_app_name
            if st.button("🔌 Connect PRAW", use_container_width=True, disabled=not can_connect_praw, key="connect_praw_btn"):
                st.session_state.fetch_attempted = False
                st.session_state.posts_df = None
                st.session_state.comments_data = {}
                st.session_state.selected_post_ids = []
                with st.spinner("Connecting to PRAW..."):
                    connect_success, status_msg = analyzer.connect_praw(cred_client_id, cred_client_secret, reddit_app_name, reddit_username)
                st.session_state.praw_connection_status = status_msg
                st.rerun()
            status_msg = st.session_state.praw_connection_status
            if "✅" in status_msg:
                st.success(status_msg)
            elif "❌" in status_msg:
                st.error(status_msg)
                if "401" in status_msg:
                    st.warning("Authentication Error (401)? **RE-VERIFY** Client ID & Secret. Ensure App Type is 'script'.", icon="🚨")
            else:
                st.info(status_msg)

        with st.expander("Groq Connection (for LLM Analysis)", expanded=not groq_connected):
            st.caption("Connect Groq account (optional) for AI comment summaries.")
            groq_api_key_input = st.text_input("Groq API Key", value=os.getenv('GROQ_API_KEY', ''), type="password", key="groq_key", help="Get from console.groq.com")
            if st.button("💡 Connect Groq", use_container_width=True, disabled=not groq_api_key_input, key="connect_groq_btn"):
                with st.spinner("Connecting to Groq..."):
                    connect_success, status_msg = analyzer.connect_groq(groq_api_key_input)
                st.session_state.groq_connection_status = status_msg
                st.rerun()
            status_msg = st.session_state.groq_connection_status
            if "✅" in status_msg:
                st.success(status_msg)
                try:
                    default_model_index = GROQ_TEXT_MODELS.index(st.session_state.selected_groq_model)
                except ValueError:
                    try:
                        default_model_index = GROQ_TEXT_MODELS.index(DEFAULT_GROQ_MODEL)
                    except ValueError:
                        default_model_index = 0
                    st.session_state.selected_groq_model = GROQ_TEXT_MODELS[default_model_index]
                st.session_state.selected_groq_model = st.selectbox(
                    "Select Groq Model:", options=GROQ_TEXT_MODELS, index=default_model_index,
                    key="groq_model_selector", help="Choose AI model.",
                )
            elif "❌" in status_msg:
                st.error(status_msg)
            else:
                st.info(status_msg)
        st.divider()

        # Subreddit Selection
        st.subheader("🎯 Target Subreddit")
        search_method = st.radio("Find By:", ["Name or URL", "Keyword Search"], horizontal=True, key="search_method", label_visibility="collapsed")
        subreddit_target_input = None
        if search_method == "Keyword Search":
            keyword = st.text_input("Keyword Search:", key="sub_keyword", disabled=not praw_connected, placeholder="Enter keyword (PRAW needed)" if not praw_connected else "Enter keyword...")
            if keyword and praw_connected:
                subreddits = analyzer.search_subreddits(keyword)
                search_error_occurred = st.session_state._search_had_error
                if subreddits:
                    subreddit_options = {f"r/{sub['name']} ({sub['subscribers']:,} subs)": sub['name'] for sub in subreddits}
                    options_list = ["-- Select from search results --"] + list(subreddit_options.keys())
                    selected_display = st.selectbox("Select subreddit:", options_list, key="sub_select", index=0)
                    if selected_display != "-- Select from search results --":
                        subreddit_target_input = subreddit_options[selected_display]
                elif not search_error_occurred:
                    st.warning(f"No subreddits found matching '{keyword}'. Try a different keyword.")
            elif keyword and not praw_connected:
                st.warning("Connect PRAW above to enable keyword search.")
            if not keyword:
                st.session_state._search_had_error = False
        else:  # Name or URL
            st.session_state._search_had_error = False
            name_or_url = st.text_input("Subreddit Name or URL:", key="sub_name_url", placeholder="e.g., 'python' or full URL")
            if name_or_url:
                extracted_name = extract_subreddit_name(name_or_url)
                if extracted_name:
                    try:
                        subreddit_target_input = validate_subreddit(extracted_name)
                    except ValueError as e:
                        st.error(f"Invalid subreddit name: {e}")
                else:
                    st.error("Invalid format. Enter name or full URL.")
        current_subreddit = st.session_state.get('subreddit_target', None)
        if subreddit_target_input and current_subreddit != subreddit_target_input:
            logger.info(f"Subreddit target changed to: r/{subreddit_target_input}")
            st.session_state.fetch_attempted = False
            st.session_state.posts_df = None
            st.session_state.comments_data = {}
            st.session_state.selected_post_ids = []
            st.session_state.subreddit_target = subreddit_target_input
            st.session_state._search_had_error = False
            st.rerun()
        st.caption(f"Selected: `r/{st.session_state.subreddit_target}`" if st.session_state.subreddit_target else "No subreddit selected")
        st.divider()

        # Post Filtering
        st.subheader("📄 Post Filters")
        filters_enabled = st.session_state.subreddit_target is not None
        post_sort_options = ['Hot', 'New', 'Top (Day)', 'Top (Week)', 'Top (Month)', 'Top (Year)', 'Top (All Time)', 'Controversial (Day)', 'Controversial (Week)', 'Controversial (Month)', 'Controversial (Year)', 'Controversial (All Time)']
        post_sort = st.selectbox("Sort Posts By:", post_sort_options, index=0, key="post_sort", help="Public fetch uses basic sort; PRAW uses timeframe.", disabled=not filters_enabled)
        date_options = ["Any Time", "Last 24 hours", "Last 7 days", "Last 30 days", "Last 90 days", "Last Year"]
        date_option = st.selectbox("Date Range:", date_options, index=0, key="date_range", help="Filters posts. PRAW respects range precisely.", disabled=not filters_enabled)
        end_date = datetime.now()
        if date_option == "Last 24 hours": start_date = end_date - timedelta(days=1)
        elif date_option == "Last 7 days": start_date = end_date - timedelta(days=7)
        elif date_option == "Last 30 days": start_date = end_date - timedelta(days=30)
        elif date_option == "Last 90 days": start_date = end_date - timedelta(days=90)
        elif date_option == "Last Year": start_date = end_date - timedelta(days=365)
        else: start_date = datetime(2005, 6, 23)
        limit = st.slider("Max Posts to Fetch:", min_value=10, max_value=150, value=25, step=5, key="post_limit", help="Max posts to retrieve.", disabled=not filters_enabled)
        if st.button("🔍 Fetch Posts", use_container_width=True, type="primary", disabled=not filters_enabled, key="fetch_posts_btn"):
            logger.info(f"Fetch Posts button clicked for r/{st.session_state.subreddit_target}")
            st.session_state.fetch_attempted = True
            st.session_state.posts_df = None
            st.session_state.comments_data = {}
            st.session_state.selected_post_ids = []
            with st.spinner(f"Fetching posts from r/{st.session_state.subreddit_target}..."):
                posts = analyzer.fetch_posts_multimethod(st.session_state.subreddit_target, post_sort, limit, start_date, end_date)
            st.session_state.posts_df = posts
            if posts is None: logger.warning("Post fetch completed: result None.")
            elif posts.empty: logger.info("Post fetch completed: result empty.")
            else: logger.info(f"Post fetch completed: Found {len(posts)} posts.")
            st.rerun()
        elif not filters_enabled:
            st.info("Select a subreddit above.")
        st.divider()

        # Cache Control & Guide Link
        st.subheader("🛠️ Options")
        if st.button("🧹 Clear Cached Data", use_container_width=True, key="clear_cache_btn", help="Clear cached Reddit/LLM data."):
            logger.info("Clear Cache button clicked.")
            st.cache_data.clear()
            st.success("Cache cleared!")
            time.sleep(2)
            st.rerun()
        st.link_button("📖 View Usage Guide (Placeholder)", "/Guide", help="Open guide page.")

    # --- Main Area Display ---
    current_subreddit = st.session_state.get('subreddit_target', None)
    fetch_attempted = st.session_state.get('fetch_attempted', False)
    posts_df = st.session_state.get('posts_df', None)

    if posts_df is not None and not posts_df.empty:
        st.header(f"📰 Fetched Posts from r/{current_subreddit}")
        st.caption(f"Displaying {len(posts_df)} posts. Select posts to analyze comments. Click headers to sort.")
        posts_display_df = posts_df.copy()
        posts_display_df.insert(0, "Select", False)
        posts_display_df['Created Date'] = posts_display_df['Created Date'].apply(format_datetime)
        posts_display_df['Sentiment Compound'] = posts_display_df['Sentiment Compound'].round(2)
        posts_display_df['Score'] = posts_display_df['Score'].astype(int)
        posts_display_df['Comments Count'] = posts_display_df['Comments Count'].astype(int)
        column_config = {
            "Select": st.column_config.CheckboxColumn(required=True, default=False, help="✔️ Select post(s)"),
            "Title": st.column_config.TextColumn(width="large", help="💬 Post title."),
            "Score": st.column_config.NumberColumn(format="%d", help="⬆️⬇️ Post score."),
            "Comments Count": st.column_config.NumberColumn(format="%d", label="Cmts", help="🗨️ Total comments."),
            "Sentiment Compound": st.column_config.NumberColumn(format="%.2f", label="Sent.", help="😊 Post sentiment."),
            "Created Date": st.column_config.TextColumn(width="small", help="📅 Post date."),
            "URL": st.column_config.LinkColumn(display_text="🔗 Link", width="small", help="Link to post."),
            "Post ID": None, "Content": None,
            "Author": st.column_config.TextColumn(width="medium", help="👤 Post author."),
            "Sentiment Polarity": None, "Sentiment Subjectivity": None,
        }
        column_order = ['Select', 'Title', 'Score', 'Comments Count', 'Sentiment Compound', 'Created Date', 'Author', 'URL']
        edited_df = st.data_editor(
            posts_display_df, column_order=column_order, column_config=column_config,
            key="post_selector", disabled=['Title', 'Score', 'Comments Count', 'Sentiment Compound', 'Created Date', 'URL', 'Author'],
            use_container_width=True, hide_index=True, num_rows="fixed",
        )
        selected_indices = edited_df[edited_df['Select']].index
        st.session_state.selected_post_ids = posts_df.loc[selected_indices, 'Post ID'].tolist()
        st.info(f"Selected **{len(st.session_state.selected_post_ids)}** post(s) for comment analysis.", icon="👇" if st.session_state.selected_post_ids else "ℹ️")
        try:
            csv_posts = posts_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Posts Data (CSV)", csv_posts, f"reddit_posts_{current_subreddit}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv", key="download_posts")
        except Exception as e:
            logger.error(f"Failed posts CSV: {e}", exc_info=True)
            st.warning("Could not generate posts download.")

        # Comment Fetching Section
        if st.session_state.selected_post_ids:
            st.divider()
            st.subheader("💬 Comment Analysis")
            st.write(f"Analyze comments for **{len(st.session_state.selected_post_ids)}** selected post(s).")
            col_com1, col_com2 = st.columns([3, 2])
            with col_com1:
                comment_sort_options = ['Top', 'New', 'Controversial', 'Q&A', 'Old']
                comment_sort = st.selectbox("Sort Comments By:", comment_sort_options, index=0, key="comment_sort", help="Selects sort order *if* using PRAW API.")
            with col_com2:
                comment_limit = st.number_input("Max Comments per Post:", min_value=5, max_value=500, value=50, step=5, key="comment_limit", help="Max comments per post.")
            analyze_button_label = f"📊 Analyze Comments for {len(st.session_state.selected_post_ids)} Post(s)"
            if st.button(analyze_button_label, type="primary", use_container_width=True, key="analyze_comments_btn"):
                logger.info(f"Analyze Comments button clicked for {len(st.session_state.selected_post_ids)} posts.")
                st.session_state.comments_data = {}
                total_posts_to_fetch = len(st.session_state.selected_post_ids)
                progress_bar = st.progress(0, text="Initializing concurrent comment fetch...")
                status_placeholder = st.empty()

                completed_count = 0

                def progress_callback(post_id: str, success: bool, count: Optional[int]):
                    nonlocal completed_count
                    completed_count += 1
                    progress_val = completed_count / total_posts_to_fetch
                    progress_bar.progress(progress_val, text=f"Fetched {completed_count}/{total_posts_to_fetch} posts...")

                comments_results = fetch_all_comments_concurrent(
                    post_ids=st.session_state.selected_post_ids,
                    subreddit_name=current_subreddit,
                    sort_by=comment_sort,
                    limit=comment_limit,
                    praw_details=analyzer.praw_details,
                )

                fetch_success_count = 0
                fetch_fail_count = 0
                for post_id, df in comments_results.items():
                    st.session_state.comments_data[post_id] = df
                    if df is not None and not df.empty:
                        fetch_success_count += 1
                        logger.info(f"Success: Fetched {len(df)} for {post_id}.")
                    else:
                        fetch_fail_count += 1
                        logger.warning(f"Failed/no comments for {post_id}.")

                progress_bar.empty()
                status_placeholder.empty()
                if fetch_success_count > 0 and fetch_fail_count == 0:
                    st.success(f"Comment fetching complete for all {fetch_success_count} post(s).")
                elif fetch_success_count > 0 and fetch_fail_count > 0:
                    st.warning(f"Comment fetch partial. Found for {fetch_success_count}, failed/none for {fetch_fail_count}. Check `app.log`.")
                else:
                    st.error(f"Comment fetching failed for all {fetch_fail_count} post(s). Check `app.log`.")
                logger.info(f"Comment fetch finished. Success: {fetch_success_count}, Fail/None: {fetch_fail_count}")
                st.rerun()

    elif fetch_attempted and posts_df is None and current_subreddit:
        st.error(f"Failed to fetch posts for r/{current_subreddit}. Reasons: Not found, private, network issue, invalid PRAW creds. See `app.log`.")
    elif fetch_attempted and posts_df is not None and posts_df.empty and current_subreddit:
        st.warning(f"No posts found for `r/{current_subreddit}` matching filters. Try adjusting filters.", icon="⚠️")

    # Display Combined Comments & Analysis
    valid_comment_dfs = [df for df in st.session_state.get('comments_data', {}).values() if df is not None and not df.empty]
    if valid_comment_dfs:
        all_comments_df = pd.concat(valid_comment_dfs, ignore_index=True)
        st.divider()
        st.header("💬 Fetched Comments (Combined)")
        st.caption(f"Displaying **{len(all_comments_df)}** comments from **{len(valid_comment_dfs)}** post(s). Filter below.")
        comment_filter = st.text_input("Filter displayed comments by keyword:", placeholder="Search comment text...", key="comment_filter")
        if comment_filter:
            filtered_comments_df = all_comments_df[all_comments_df['Comment Body'].astype(str).str.contains(comment_filter, case=False, na=False)]
            st.caption(f"Showing **{len(filtered_comments_df)}** comments matching: '{comment_filter}'.")
        else:
            filtered_comments_df = all_comments_df
        if not filtered_comments_df.empty:
            comments_display_df = filtered_comments_df.copy()
            comments_display_df['Created Date'] = comments_display_df['Created Date'].apply(format_datetime)
            comments_display_df['Sentiment Compound'] = comments_display_df['Sentiment Compound'].round(2)
            comments_display_df['Score'] = comments_display_df['Score'].astype(int)
            comment_column_config = {
                "Post ID": st.column_config.TextColumn(width="small", help="🔗 Original post ID."),
                "Comment Body": st.column_config.TextColumn(width="large", help="💬 Comment text."),
                "Score": st.column_config.NumberColumn(format="%d", help="⬆️⬇️ Score."),
                "Sentiment Compound": st.column_config.NumberColumn(format="%.2f", label="Sent.", help="😊 Sentiment."),
                "Created Date": st.column_config.TextColumn(width="small", help="📅 Date."),
                "Author": st.column_config.TextColumn(width="medium", help="👤 Commenter."),
                "Comment ID": None, "Is Submitter": None, "Sentiment Polarity": None, "Sentiment Subjectivity": None,
            }
            column_order_comments = ['Post ID', 'Comment Body', 'Score', 'Sentiment Compound', 'Created Date', 'Author']
            st.dataframe(comments_display_df[column_order_comments], column_config=comment_column_config, use_container_width=True, hide_index=True)
            try:
                csv_comments = filtered_comments_df.to_csv(index=False).encode('utf-8')
                download_filename = f"reddit_comments_{current_subreddit}"
                if comment_filter:
                    download_filename += f"_filtered_{comment_filter.replace(' ', '_')[:20]}"
                download_filename += f"_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                st.download_button("📥 Download Displayed Comments (CSV)", csv_comments, download_filename, "text/csv", key="download_comments_filtered")
            except Exception as e:
                logger.error(f"Failed comments CSV: {e}", exc_info=True)
                st.warning("Could not generate comments download.")
        else:
            st.info("No comments match the current filter.")

        st.divider()
        st.header("📊 Visualizations")
        st.caption("Visualizations based on comments displayed above (respects filter).")
        if not filtered_comments_df.empty:
            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                plot_sentiment_distribution(filtered_comments_df, 'Sentiment Compound')
            with viz_col2:
                generate_word_cloud(filtered_comments_df['Comment Body'], "Comment Word Cloud")
        else:
            st.info("No comments displayed to visualize.")

        st.divider()
        st.header("🤖 LLM Analysis (Optional)")
        if not GROQ_AVAILABLE:
            st.info("Install `groq` (`pip install groq`) for LLM.", icon="ℹ️")
        elif not groq_connected:
            st.warning("Connect Groq in the sidebar for LLM.", icon="💡")
        elif filtered_comments_df.empty:
            st.info("No comments displayed to analyze.", icon="💬")
        else:
            enable_llm = st.toggle("Enable LLM Analysis Section", value=False, key="llm_toggle", help="Toggle LLM options.")
            if enable_llm:
                selected_model = st.session_state.get('selected_groq_model', DEFAULT_GROQ_MODEL)
                llm_focus_options = ["Overall Summary", "Themes and Topics", "Sentiment Analysis (Detailed)", "Pain Points / Complaints", "Positive Feedback / Praise", "Actionable Insights / Suggestions"]
                analysis_type = st.selectbox("Select LLM Analysis Focus:", llm_focus_options, key="llm_focus", index=0, help="Choose analysis angle.")
                if st.button(f"🧠 Run '{analysis_type}' Analysis with {selected_model}", type="secondary", use_container_width=True, key="run_llm_btn"):
                    if not filtered_comments_df.empty:
                        with st.spinner(f"Generating LLM analysis ({analysis_type})..."):
                            llm_summary = analyzer.generate_llm_analysis(filtered_comments_df, analysis_type, selected_model)
                        if llm_summary:
                            if "LLM Error:" in llm_summary or "LLM Prep Error:" in llm_summary:
                                st.error(llm_summary)
                            elif "LLM Info:" in llm_summary:
                                st.info(llm_summary)
                            else:
                                st.subheader(f"🤖 LLM Analysis Results ({analysis_type})")
                                # Security fix: sanitize LLM output before rendering to prevent XSS
                                st.markdown(sanitize_markdown(llm_summary))
                        else:
                            st.error("LLM analysis returned no result. Check logs.")
                    else:
                        st.warning("No comments available to analyze.")
            else:
                st.caption("Toggle switch above to run AI analysis.")

    # Initial Guidance
    elif not current_subreddit:
        st.info("👈 **Welcome!** Select a subreddit in the sidebar to begin.")
    elif current_subreddit and posts_df is None and not fetch_attempted:
        st.info(f"🚀 Ready to fetch posts from `r/{current_subreddit}`. Click **'🔍 Fetch Posts'**.")

    # Footer
    st.divider()
    st.markdown("""
    <style>.footer{font-size: 0.875rem;color: #808495;text-align: center;padding-top: 2rem;padding-bottom: 1rem;border-top: 1px solid #e1e4eb;}.footer a{text-decoration: none;color: #FF4500;font-weight: 500;}.footer a:hover{text-decoration: underline;color: #FF5700;}.footer img{margin-left: 8px;margin-right: 3px;vertical-align: middle;height: 18px;width: 18px;opacity: 0.8;filter: grayscale(30%);transition: opacity 0.2s ease-in-out, filter 0.2s ease-in-out;}.footer a:hover img{opacity: 1.0;filter: grayscale(0%);}</style>
    <div class="footer">Created by <a href="https://www.linkedin.com/in/tayeebkhan/" target="_blank" title="Visit Tayeeb's LinkedIn">Tayeeb Khan</a><a href="https://github.com/followtayeeb" target="_blank" title="Visit Tayeeb's GitHub"><img src="https://img.icons8.com/material-outlined/24/808495/github.png" alt="GitHub"/></a><a href="https://www.linkedin.com/in/tayeebkhan/" target="_blank" title="Visit Tayeeb's LinkedIn"><img src="https://img.icons8.com/material-outlined/24/808495/linkedin.png" alt="LinkedIn"/></a></div>
    """, unsafe_allow_html=True)


# --- Run App ---
if __name__ == "__main__":
    main()

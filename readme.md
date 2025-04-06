# Reddit Voice of Customer (VOC) Analyzer

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
<!-- Add License badge later if you make it public -->
<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) -->

A Streamlit web application designed to fetch, analyze, and visualize posts and comments from Reddit subreddits. It helps understand community sentiment, key topics, and feedback patterns using data analysis and optional Large Language Model (LLM) integration.

The application employs a hybrid approach:
*   It first attempts to fetch data using **public scraping methods** (via `requests` on `old.reddit.com`) which doesn't require authentication but might be less reliable or limited.
*   It uses the **PRAW (Python Reddit API Wrapper)** library as a more robust fallback and for features like subreddit searching and specific comment sorting, which requires user-provided Reddit API credentials.

<!-- Optional: Add a screenshot/gif here -->
<!-- ![App Screenshot](link/to/your/screenshot.png) -->

## Key Features

*   **Subreddit Selection:**
    *   Specify target subreddit by **Name** (e.g., `learnpython`).
    *   Specify target subreddit by **URL** (e.g., `https://www.reddit.com/r/learnpython`).
    *   **Search** for subreddits by keyword (requires PRAW connection).
*   **Post Fetching:**
    *   Fetch posts based on sorting criteria (Hot, New, Top, Controversial) with time filters (Day, Week, Month, Year, All Time).
    *   Filter posts by date range (e.g., Last 30 days, Last Year).
    *   Specify the maximum number of posts to fetch.
*   **Comment Fetching:**
    *   Select fetched posts to retrieve their comments.
    *   Specify maximum comments per post.
    *   Utilizes PRAW (if connected) for reliable comment fetching and sorting (Top, New, Controversial, etc.). Falls back to public scraping.
*   **Data Display & Filtering:**
    *   View fetched posts and comments in interactive, sortable tables.
    *   Filter displayed comments by keywords.
*   **Analysis & Visualization:**
    *   **Sentiment Analysis:** Calculates polarity, subjectivity (TextBlob), and compound sentiment (NLTK VADER) for posts and comments.
    *   **Sentiment Distribution:** Displays a histogram showing the spread of comment sentiment scores.
    *   **Word Cloud:** Generates a word cloud from the displayed comment text to highlight frequent terms.
*   **LLM Integration (Optional):**
    *   Connect to the **Groq API** for fast AI-powered analysis.
    *   Select different analysis focuses (Overall Summary, Themes, Sentiment Detail, Pain Points, Praise, Actionable Insights).
    *   Choose from various compatible Groq language models.
*   **Data Export:**
    *   Download fetched posts data as a CSV file.
    *   Download displayed (potentially filtered) comments data as a CSV file.
*   **Caching:** Caches API responses and LLM results to improve performance and reduce API calls. Cache can be cleared manually via the sidebar.
*   **Logging:** Logs application activity and errors to `app.log` for debugging.

## Tech Stack

*   **Language:** Python 3.9+
*   **Web Framework:** Streamlit
*   **Reddit API:** PRAW (Python Reddit API Wrapper)
*   **Public Scraping:** Requests
*   **Data Handling:** Pandas
*   **NLP/Sentiment:** TextBlob, NLTK (VADER, Stopwords)
*   **Visualization:** Plotly Express, Matplotlib, WordCloud
*   **LLM API:** Groq Python SDK
*   **Environment:** python-dotenv

## Setup Instructions

1.  **Prerequisites:**
    *   Python 3.9 or higher installed.
    *   Git installed and configured in your PATH (see Troubleshooting below if `git` command fails).

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/[Your GitHub Username]/reddit-voc-analyzer.git
    cd reddit-voc-analyzer
    ```
    *(Replace `[Your GitHub Username]` with your actual username)*

3.  **(Recommended) Create and Activate a Virtual Environment:**
    ```bash
    # Create environment (use python3 on macOS/Linux if needed)
    python -m venv venv

    # Activate (Windows PowerShell)
    .\venv\Scripts\Activate.ps1

    # Activate (Windows Command Prompt)
    .\venv\Scripts\activate.bat

    # Activate (macOS/Linux Bash/Zsh)
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download NLTK Data:**
    The application attempts to download necessary NLTK data (`vader_lexicon`, `stopwords`) on first run if not found. If this fails due to network issues or permissions, you can download them manually:
    ```bash
    python -m nltk.downloader vader_lexicon stopwords
    ```

6.  **Create and Configure `.env` File:**
    *   Copy the example file:
        ```bash
        # Windows
        copy .env.example .env
        # macOS/Linux
        cp .env.example .env
        ```
    *   **Edit the new `.env` file** and fill in your credentials:
        *   `REDDIT_CLIENT_ID`: Your Reddit script app's Client ID.
        *   `REDDIT_CLIENT_SECRET`: Your Reddit script app's Secret.
        *   `REDDIT_APP_NAME`: The **exact name** you gave your app on Reddit.
        *   `REDDIT_USERNAME` (Optional): Your Reddit username (without `/u/`).
        *   `GROQ_API_KEY` (Optional): Your API key from [console.groq.com](https://console.groq.com/) if you want to use LLM features.
    *   **Get Reddit Credentials:** Create a new application of type **'script'** on [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps). You will find the Client ID (under the app name) and the Secret there.

## Running the Application

1.  Make sure your virtual environment (if created) is activated.
2.  Navigate to the project directory in your terminal.
3.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
4.  Open the URL shown in your terminal (usually `http://localhost:8501`) in your web browser.
5.  Use the sidebar to configure PRAW/Groq connections and select a subreddit to begin analysis.

## Troubleshooting

*   **`git` command not found:** See Step 1 of Setup - ensure Git is installed and its `cmd` directory is in your system's PATH environment variable. Restart your terminal after making PATH changes.
*   **PRAW Connection Errors (401 Unauthorized):** This almost always means your `REDDIT_CLIENT_ID` or `REDDIT_CLIENT_SECRET` in the `.env` file (or entered in the UI) is incorrect. Double-check them against [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps). Also ensure the app type on Reddit is 'script'.
*   **PRAW Connection Errors (Other):** Network issues, Reddit API downtime, or incorrect App Name might cause other errors. Check the status message in the app and the `app.log` file.
*   **NLTK Data Errors:** If sentiment analysis or word clouds fail with `LookupError`, try running the manual NLTK download command from Step 5 of Setup.
*   **Other Errors:** Check the `app.log` file in the project directory for detailed error messages and tracebacks.

## Future Enhancements (Ideas)

*   More sophisticated NLP analysis (Topic Modeling, NER).
*   User authentication for saving settings (requires more complex setup).
*   Support for analyzing user profiles or specific post URLs directly.
*   More visualization options.
*   Error handling improvements.

## License

Currently, this repository is private. If made public in the future, an appropriate open-source license (e.g., MIT) will be added.

*(If you make the repo public later, uncomment the MIT license badge at the top and add a `LICENSE` file with the MIT license text.)*

## Author

*   **Tayeeb Khan**
    *   GitHub: [@followtayeeb](https://github.com/followtayeeb)
    *   LinkedIn: [Tayeeb Khan](https://www.linkedin.com/in/tayeebkhan/)
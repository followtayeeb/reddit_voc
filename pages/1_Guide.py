import streamlit as st

st.set_page_config(page_title="VOC Analyzer Guide", page_icon="📖", layout="wide")

# --- Header ---
col1, col2 = st.columns([1, 8])
with col1:
    st.image("https://img.icons8.com/color/96/000000/reddit.png", width=60)
with col2:
    st.title("📖 Reddit VOC Analyzer - User Guide")
    st.caption("Understand customer feedback, sentiment, and topics from Reddit.")

st.markdown("---")

# --- Navigation / Table of Contents (using expanders) ---

with st.expander("🚀 Getting Started & API Setup", expanded=True):
    st.header("1. Initial Setup (API Keys)")
    st.info("API keys provide more reliable data access than public methods, especially for comments.")

    st.subheader("PRAW (Reddit API)")
    st.markdown("""
    *   **Why needed?** **Essential for fetching comments reliably.** Also enables keyword search and acts as a backup for fetching posts.
    *   **How to get keys:**
        1.  Log in to Reddit -> [App Preferences](https://www.reddit.com/prefs/apps).
        2.  Scroll down -> "are you a developer? create an app...".
        3.  Fill form: Name (e.g., `MyVOCApp`), **Type `script`** (Crucial!), Redirect URI `http://localhost:8080`.
        4.  Click "Create app".
        5.  Copy the **Client ID** (e.g., `StGFM...`) and **Client Secret** (e.g., `q_M5Z...`). **Keep Secret confidential!**
    *   **Using in the App:** Paste ID & Secret in sidebar -> "PRAW Connection" -> Click "🔌 Connect PRAW". Look for ✅.
    *   **Troubleshooting:** If connection fails (esp. `ReadOnlyAuthorizer` error):
        *   **Confirm App Type:** Double-check it's set to `script` on the Reddit app page.
        *   **Verify Credentials:** Ensure ID/Secret are copied *exactly*.
        *   **Recreate:** If unsure, delete the app on Reddit and create a *new* one (select `script`!), then use the *new* credentials.
    """)

    st.subheader("Groq (Optional AI Analysis)")
    st.markdown("""
    *   **Why needed?** Powers the optional "LLM Analysis" section for AI summaries.
    *   **How to get key:** Create account at [GroqCloud](https://console.groq.com/) -> "API Keys" -> Create & copy key (`gsk_...`). **Save securely.**
    *   **Using in the App:** Paste key in sidebar -> "Groq Connection" -> Click "💡 Connect Groq". Look for ✅.
    """)
    st.caption("Tip: Store keys in `.env` file for automatic loading (see README/code).")


with st.expander("⚙️ Using the Analyzer - Step-by-Step"):
    st.header("2. Step-by-Step Usage")
    st.subheader("A. Select Subreddit")
    st.markdown("""
    1.  **Find By:** Choose `Name or URL` or `Keyword Search` (PRAW needed).
    2.  Enter the name/URL/keyword.
    3.  Select from dropdown if using search.
    4.  The sidebar confirms your selection (`Selected: r/...`). Filters below become active.
    """)

    st.subheader("B. Filter & Fetch Posts")
    st.markdown("""
    1.  **Sort Posts By:** How to initially sort posts (e.g., `Hot`, `Top (Week)`).
    2.  **Date Range:** Time window for posts.
    3.  **Max Posts:** Max posts to retrieve *after* filtering.
    4.  Click **"🔍 Fetch Posts"**. App tries public fetch, then PRAW if needed/connected.
    """)

    st.subheader("C. Analyze Posts Table")
    st.markdown("""
    *   Fetched posts appear here. Click headers to sort. Check boxes to select for comment analysis.
    """)
    # Detailed Column Explanations for Posts
    st.subheader("Understanding the Posts Table Columns:")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**`Select`**: ✔️ Check this box to include the post when analyzing comments later.")
        st.markdown("**`Title`**: 💬 The headline written by the post's author.")
        st.markdown("**`Score`**: ⬆️⬇️ **(Sortable)** Post 'karma'. Upvotes minus downvotes. Higher scores suggest more community interaction/visibility.")
        st.markdown("**`Cmts`**: 🗨️ **(Sortable)** Total number of comments. High numbers indicate active discussion.")
    with cols[1]:
        st.markdown("**`Sent.`**: 😊 **(Sortable)** Overall sentiment score (-1 to +1) of the post's title and body text combined. Helps gauge initial tone.")
        st.markdown("**`Created Date`**: 📅 When the post was submitted.")
        st.markdown("**`🔗 Link`**: Direct link to the post on Reddit.")
    st.markdown("*Download the CSV to see these additional columns:*")
    st.markdown("**`Post ID`**: Unique code identifying the post on Reddit.")
    st.markdown("**`Content`**: The main text body of the post (if provided by the author).")
    st.markdown("**`Author`**: 👤 Reddit username of the post's creator (OP - Original Poster).")
    st.markdown("**`Sentiment Polarity`**: Technical score (-1 to +1) indicating positive/negative leaning from one analysis method.")
    st.markdown("**`Sentiment Subjectivity`**: Technical score (0 to 1) indicating how objective (0) or opinionated (1) the text is.")
    st.markdown("""
    *   **Download Posts:** Click "📥 Download Posts Data (CSV)" to save this table.
    """)


    st.subheader("D. Analyze Comments")
    st.markdown("""
    1.  **Select Posts:** Ensure posts are selected above.
    2.  **Comment Options:** Set max comments per post. `Sort Comments By` only affects PRAW fetch.
    3.  Click **"💬 Analyze Comments..."**. App tries public fetch first (limited), then PRAW if connected (better).
    """)
    st.subheader("Understanding the Comments Table Columns:")
    st.markdown("""
    *   Displays combined comments from *all* selected posts. Filter using the search box. Click headers to sort.
    """)
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**`Post ID`**: 🔗 Identifier linking the comment back to its original post in the table above.")
        st.markdown("**`Comment Body`**: 💬 The actual text of the user's comment.")
        st.markdown("**`Score`**: ⬆️⬇️ **(Sortable)** Popularity of the comment itself. High score often means agreement or visibility.")
    with cols[1]:
        st.markdown("**`Sent.`**: 😊 **(Sortable)** Sentiment score (-1 to +1) for *this specific comment's text*.")
        st.markdown("**`Created Date`**: 📅 When the comment was posted.")
        st.markdown("**`Author`**: 👤 Username of the commenter.")
    st.markdown("*Download the CSV to see these additional columns:*")
    st.markdown("**`Comment ID`**: Unique code identifying this specific comment.")
    st.markdown("**`Is Submitter`**: Indicates if this commenter is also the Original Poster (OP) of the post.")
    st.markdown("**`Sentiment Polarity`**: Technical sentiment score (-1 to +1) for the comment.")
    st.markdown("**`Sentiment Subjectivity`**: Technical score (0 to 1) indicating objective vs. opinionated comment text.")
    st.markdown("""
    *   **Download Comments:** Click "📥 Download Displayed Comments (CSV)" to save the currently shown (filtered) comments.
    """)


    st.subheader("E. Visualizations")
    st.markdown("""
    *   Below comments: **Sentiment Distribution** (histogram of comment scores) and **Word Cloud** (frequent terms). Download the cloud image.
    """)

    st.subheader("F. LLM Analysis (Optional)")
    st.markdown("""
    *   **Requires Groq Connection.** Toggle **Enable**. Choose **Focus**. Click **"🧠 Run LLM Analysis..."**.
    *   AI analyzes *displayed comments* for themes/sentiment/insights. Results appear below.
    """)

    st.subheader("G. Cache")
    st.markdown("""
    *   ♻️ Fetched data is temporarily saved (cached) for ~1 hour. Use **"🧹 Clear Cache"** in sidebar for immediate fresh data.
    """)

with st.expander("📊 Interpreting the Results"):
    st.header("3. Interpreting Results")
    st.subheader("Sentiment Scores (-1.0 to +1.0)")
    st.markdown("""
    *   `< -0.05`: Leaning **Negative**. Look for complaints, pain points, dissatisfaction.
    *   `-0.05 to +0.05`: **Neutral**. Could be factual, mixed, or unclear. Needs reading.
    *   `> +0.05`: Leaning **Positive**. Look for praise, satisfaction, suggestions, agreement.
    *   *Intensity:* Scores near -1 or +1 are stronger. Scores near 0 are weaker.
    *   *Context is Key:* Automated analysis isn't perfect (sarcasm!). **Always read comments** alongside scores.
    """)
    st.subheader("Score / Comment Count")
    st.markdown("""
    *   **High `Score`:** Indicates strong community reaction (positive or negative). These posts/comments are often significant.
    *   **High `Comments Count`:** Signifies topics that generated lots of discussion. Dive into these for diverse opinions.
    """)
    st.subheader("Visuals & LLM")
    st.markdown("""
    *   ☁️ **Word Cloud:** Quick glance at frequently mentioned words. What stands out?
    *   🤖 **LLM Analysis:** Provides a starting point summary. Use its themes/pain points to focus your reading of the actual comments. Verify AI conclusions.
    """)

with st.expander("❓ Troubleshooting"):
    st.header("4. Troubleshooting")
    st.markdown("""
    *   **PRAW Connection Failed / `ReadOnlyAuthorizer` Error:**
        *   **Top Priority:** Go to [Reddit App Preferences](https://www.reddit.com/prefs/apps). **Confirm** the `Type` for your app is **`script`**. If not, DELETE it and CREATE a NEW one, selecting `script`. Use the **new** Client ID/Secret.
        *   **Second:** Carefully **re-verify** the Client ID and Secret in the sidebar match *exactly* what's shown on Reddit. No typos, no extra spaces. Copy/paste again.
        *   Check your Reddit account status (verified email, no bans/restrictions).
        *   Check the `app.log` file (in the same folder as the script) for detailed error messages.
    *   **Fetch Posts/Comments Failed:** Check subreddit name (case-sensitive sometimes). Subreddit could be private/banned. Reddit might have temporary issues. Rate limits hit (wait or use PRAW if possible). Check `app.log`.
    *   **LLM Fails:** Check Groq API key & connection status in sidebar. Check your Groq account usage/limits.
    *   **Slow / Old Data:** Click "🧹 Clear Cache" in sidebar.
    *   **General Errors:** The `app.log` file is your best friend! It contains detailed error info.
    """)

# --- Credits ---
st.divider()
st.header("Credits & Contact")
st.markdown("""
*   **App created by:** Tayeeb Khan
*   **Contact / Links:**
    *   [![GitHub](https://img.icons8.com/material-outlined/24/000000/github.png)](https://github.com/followtayeeb) [GitHub](https://github.com/followtayeeb)
    *   [![LinkedIn](https://img.icons8.com/material-outlined/24/000000/linkedin.png)](https://www.linkedin.com/in/tayeebkhan/) [LinkedIn](https://www.linkedin.com/in/tayeebkhan/)
""")
st.caption("Libraries used: Streamlit, PRAW, Requests, Pandas, TextBlob, NLTK, Plotly, WordCloud, Matplotlib, Groq API.")
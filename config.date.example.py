"""
Configuration file template for Twitter API crawler

INSTRUCTIONS:
1. Copy this file to config.py: cp config.example.py config.py
2. Edit config.py and add your API key
3. Adjust other settings as needed
"""

# API Configuration
API_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"
API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key

# Crawling Configuration
TWEETS_PER_DATE = 50000
TWEETS_PER_PAGE = 20
CALLS_PER_DATE = TWEETS_PER_DATE // TWEETS_PER_PAGE  # 2500 calls

# File paths configuration
STATE_FILE = "crawler_state.json"
LOG_FILE = "tweet_crawler.log"
DATA_DIR = "tweet_data"
TEST_OUTPUT_FILE = "test.jsonl"
PARQUET_OUTPUT_DIR = "parquet_data"
SENTIMENT_OUTPUT_DIR = "sentiment_results"
# Batch processing configuration
BATCH_MAX_LINES = 20  # Max requests per batch file
BATCH_PROMPT_ID = "YOUR_PROMPT_ID"  # Replace with your prompt id
BATCH_PROMPT_VERSION = "latest"  # Replace with your prompt version
BATCH_MODEL = "gpt-4.1"  # Model used for batch responses
BATCH_LIST_FILE = "sentiment_results/batch_list.json"

# Keywords to search for
KEYWORDS = [
    "AI",
    "artificial intelligence",
    "AGI",
    "LLM",
    "large language model",
    "genAI",
    "anti-AI",
    "OpenAI",
    "Anthropic",
    "Nvidia",
    "NVDA",
    "Cursor",
    "Mistral",
    "Perplexity",
    "GPT",
    "Claude",
    "Gemini",
    "Grok",
    "Llama",
    "Deepseek",
    "Qwen",
    "Doubao",
    "Stable Diffusion",
    "Midjourney",
    "Codex",
    "CharacterAI",
    "ChatGPT",
    "Geoffery Hinton",
    "Copilot",
    "Sam Altman",
    "Feifei Li",
    "Andrej Karpathy",
    "LeCun",
    "RLHF",
    "vibe coding",
]

# Date range configuration
START_DATE = "2024-03-01"  # Format: YYYY-MM-DD
END_DATE = "2025-03-20"  # Format: YYYY-MM-DD
TARGET_DAYS = [1, 10, 20]  # Days of the month to crawl

# Search configuration
LANGUAGE = "en"  # Language filter for tweets
QUERY_TYPE = "Latest"  # "Latest" or "Top"

# Rate limiting
REQUEST_DELAY = 0.5  # Seconds to wait between API requests

# OpenAI Configuration (for sentiment analysis)
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE"  # Replace with your OpenAI API key
OPENAI_MODEL = "gpt-4.1"  # Model to use for sentiment analysis
SENTIMENT_SAMPLE_SIZE = 1000  # Number of tweets to sample for sentiment analysis

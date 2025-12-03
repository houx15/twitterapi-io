"""
Configuration file template for Twitter API crawler

INSTRUCTIONS:
1. Copy this file to config.py: cp config.example.py config.py
2. Edit config.py and add your API key
3. Adjust other settings as needed
"""

# API Configuration
API_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"
API_KEY = "YOUR_API_KEY_HERE"


# Crawling Configuration
TWEETS_PER_DATE = None
TWEETS_PER_PAGE = 20
CALLS_PER_DATE =  None
FORCE_CALL =  False
MAX_ID_RETRY_COUNT = 2
DEFAULT_SINCE = None
DEFAULT_UNTIL = "2020-12-31"

# File paths configuration
BASE_FOLDER = "YOUR_BASE_FOLDER_HERE"
STATE_FILE = f"{BASE_FOLDER}/crawler_state.json"
LOG_FILE = f"{BASE_FOLDER}/tweet_crawler.log"
DATA_DIR = f"{BASE_FOLDER}/tweet_data"
TEST_OUTPUT_FILE = f"{BASE_FOLDER}/test.jsonl"
PARQUET_OUTPUT_DIR = f"{BASE_FOLDER}/parquet_data"
SENTIMENT_OUTPUT_DIR = f"{BASE_FOLDER}/sentiment_results"

USERNAME_FILE = "YOUR_USERNAME_FILE_HERE"
PROCESSED_FILE = f"{BASE_FOLDER}/processed_ids.txt"

# Keywords to search for
KEYWORDS = [
    "keywords"
]

# Date range configuration
START_DATE = None
END_DATE = None
TARGET_DAYS = None


# Search configuration
LANGUAGE = "en"  # Language filter for tweets
QUERY_TYPE = "Latest"  # "Latest" or "Top"

# Rate limiting
REQUEST_DELAY = 0.01  # Seconds to wait between API requests

# OpenAI Configuration (for sentiment analysis)
OPENAI_API_KEY = "sk-proj-xxxxxxxxxxxxxxxxxx"  # Replace with your OpenAI API key
OPENAI_MODEL = "gpt-5.1"  # Model to use for sentiment analysis
SENTIMENT_SAMPLE_SIZE = 1000  # Number of tweets to sample for sentiment analysis

# Batch API Configuration
BATCH_MAX_LINES = 50000  # Maximum lines per batch file
BATCH_PROMPT_ID = "pmpt_xxxxxxxxxxxxxx"
BATCH_PROMPT_VERSION = "2"
BATCH_MODEL = "gpt-5-mini"  # Model to use for batch API
BATCH_LIST_FILE = "YOUR_BATCH_LIST_FILE_HERE"  # File to track all batches
"""
Twitter API crawler using twitterapi.io

This script crawls tweets based on specific keywords and date ranges.
It supports test mode and formal crawl mode using the fire package.

"""

import requests
import json
import os
import logging
import importlib
import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import fire
import time
import random

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2
REQUEST_TIMEOUT = 30

logger = logging.getLogger(__name__)


def load_config(config_name: str = "config"):
    """
    Dynamically load configuration module.
    
    Args:
        config_name: Name of the config module or path to config file (default: "config")
                     Can be:
                     - Module name: "config", "config_custom", etc.
                     - File path: "config.xxx.py", "configs/xxx.py", "./configs/config.xxx.py", etc.
                     - Absolute path: "/path/to/config.xxx.py"
    
    Returns:
        Config module object
    """
    # Check if it's a file path (contains .py or /)
    if config_name.endswith('.py') or '/' in config_name or '\\' in config_name:
        # Load from file path
        config_path = Path(config_name)
        if not config_path.is_absolute():
            # If relative path, try to find it relative to the script directory
            script_dir = Path(__file__).parent
            config_path = script_dir / config_name
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Generate a unique module name based on file path
        module_name = f"config_{config_path.stem}_{hash(str(config_path))}"
        
        # Load module from file
        spec = importlib.util.spec_from_file_location(module_name, config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create spec for config file: {config_path}")
        
        config_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = config_module
        spec.loader.exec_module(config_module)
        
        logger.info(f"Loaded configuration from file: {config_path}")
        return config_module
    else:
        # Load as module name
        try:
            config_module = importlib.import_module(config_name)
            logger.info(f"Loaded configuration from module: {config_name}")
            return config_module
        except ImportError as e:
            logger.error(f"Failed to import config module '{config_name}': {e}")
            raise


def setup_logging(config):
    """Setup logging with configuration from config module."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler()
        ],
    )


def append_tweets(output_file: Path, tweets: List[Dict[str, Any]]):
    """Append tweets as JSON lines, one per line."""
    if not tweets:
        return
    with open(output_file, "a") as f:
        for tweet in tweets:
            f.write(json.dumps(tweet))
            f.write("\n")

def build_query(
    keywords: List[str],
    username: str = None,
) -> str:
    """
    Build the query string for Twitter API.

    Args:
        keywords: List of keywords to search for
        username: Username to search for
    Returns:
        Query string formatted for Twitter API
    """
    keyword_query = " OR ".join([f'"{kw}"' if " " in kw else kw for kw in keywords])

    basic_query = f"({keyword_query}) (from:{username}) lang:en until:2020-12-31"
    return basic_query


def load_target_ids(config) -> set[str]:
    # load all usernames as set
    with open(config.USERNAME_FILE, "r") as f:
        all_usernames = set(f.read().splitlines())
    # load all processed usernames as set
    processed_usernames = set()
    if os.path.exists(config.PROCESSED_FILE):
        with open(config.PROCESSED_FILE, "r") as f:
            processed_usernames = set(f.read().splitlines())
    return all_usernames - processed_usernames

def save_finished(config, username: str):
    with open(config.PROCESSED_FILE, "a") as f:
        f.write(username + "\n")


def crawl_tweets(
    config,
    username: str,
    output_file: Path,
) -> Dict[str, Any]:
    """
    Crawl tweets for a specific date.

    Args:
        config: Configuration module
        username: Username to crawl tweets
    Returns:
        Dictionary with crawl results
    """

    base_query = build_query(config.KEYWORDS, username)
    cursor = ""
    headers = {"X-API-Key": config.API_KEY}
    completed = False
    logger.info(
        f"Starting crawl for {username}"
    )

    while not completed:
        params = {"queryType": config.QUERY_TYPE, "query": base_query}
        # print(params)
        # Make API request
        if cursor:
            params["cursor"] = cursor
        attempt = 0
        data = None
        while attempt < MAX_RETRIES:
            try:
                response = requests.get(
                    config.API_URL,
                    headers=headers,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()
                break
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response else None
                if status == 429 or (status and status >= 500):
                    attempt += 1
                    logger.warning(
                        f"HTTP error ({status}) for {username}, retry {attempt}/{MAX_RETRIES}"
                    )
                    time.sleep(RETRY_DELAY)
                    continue
                logger.error(f"HTTP error for {username}: {e}")
                break
            except (
                requests.exceptions.RequestException,
                json.JSONDecodeError,
            ) as e:
                attempt += 1
                if attempt < MAX_RETRIES:
                    logger.warning(
                        f"Request/JSON error for {username}: {e} (retry {attempt}/{MAX_RETRIES})"
                    )
                    time.sleep(RETRY_DELAY)
                    continue
                logger.error(f"Request/JSON error for {username}: {e}")
                break

        if data is None:
            logger.error(
                f"Failed to fetch data for {username} after {MAX_RETRIES} retries, skipping"
            )
            break
        if "error" in data:
            logger.error(
                f"API error for {username}: {data.get('message', 'Unknown error')}"
            )
            break

        # Extract tweets
        tweets = data.get("tweets", [])

        # Check if there are more pages
        has_next_page = data.get("has_next_page", False)
        cursor = data.get("next_cursor")
        print("next", has_next_page, cursor)

        # Update max_id from the last tweet if we have tweets
        if tweets:
            author_username = tweets[0].get("author", {}).get("userName")
            if author_username != username:
                logger.warning(f"Skipping tweet from {author_username} for {username}")
                # completed = True
        else:
            completed = True
            logger.warning(f"No tweets found for {username}")

        append_tweets(output_file, tweets)

        if not cursor:
            completed = True


        if completed:
            # save processed
            save_finished(config, username)
            break

        # Small delay to avoid rate limiting
        time.sleep(config.REQUEST_DELAY)


    return {
        "username": username,
        "finished": completed,
    }

class RollingFile:
    def __init__(self, base_dir, prefix, max_lines=50000):
        self.base_dir = Path(base_dir)
        self.prefix = prefix
        self.max_lines = max_lines
        self.index = 0

    def _count_lines(self, file_path: Path) -> int:
        """统计文件行数"""
        if not file_path.exists():
            return 0
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def get_current_file(self) -> Path:
        """获取当前应该使用的文件名"""
        # 找到第一个不存在的文件或行数小于max_lines的文件
        while True:
            name = f"{self.prefix}_{self.index:06d}.jsonl"
            path = self.base_dir / name
            if not path.exists():
                return path
            # 如果文件存在，检查行数
            line_count = self._count_lines(path)
            if line_count < self.max_lines:
                return path
            # 如果行数>=max_lines，使用下一个文件
            self.index += 1



def crawl_username(topic: str = "abortion"):
    """
    Formal crawl mode: Crawl all dates with full configuration.
    Saves each date to a JSONL file (one tweet per line) and persists after each response.
    Maintains state file for resuming interrupted crawls.
    
    Args:
        topic: Topic to crawl (default: "abortion")
    """
    config_name = f"configs/{topic}.py"
    config = load_config(config_name)
    setup_logging(config)
    logger.info(f"=== Starting {topic} CRAWL mode ===")

    usernames = load_target_ids(config)

    # Create data directory
    data_dir = Path(config.DATA_DIR)
    data_dir.mkdir(exist_ok=True)

    rolling_file = RollingFile(data_dir, "tweets")

    with tqdm(
        total=len(usernames), initial=0, desc="Overall Progress"
    ) as overall_pbar:
        for username in usernames:
            output_file = rolling_file.get_current_file()
            result = crawl_tweets(
                config,
                username=username,
                output_file=output_file,
            )

            if result["finished"]:
                overall_pbar.update(1)

    logger.info(f"=== {topic} CRAWL mode completed ===")


if __name__ == "__main__":
    fire.Fire({"crawl": crawl_username})

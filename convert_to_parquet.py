"""
Convert crawled JSONL files to Parquet format.

This script reads tweet data from JSONL files and converts them to Parquet
format with specified columns for efficient storage and analysis.

Usage:
    # Convert all JSONL files in the data directory
    python convert_to_parquet.py convert

    # Convert a specific file
    python convert_to_parquet.py convert --input_file tweet_data/tweets_2025-03-20.jsonl

    # Convert with custom output directory
    python convert_to_parquet.py convert --output_dir parquet_data
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, date
import pandas as pd
from tqdm import tqdm
import fire

from config import DATA_DIR, PARQUET_OUTPUT_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def _to_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return str(value)
    except Exception:
        return None


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _to_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.lower()
        if lower in ("true", "yes", "1"):
            return True
        if lower in ("false", "no", "0"):
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def extract_tweet_data(tweet: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract required columns from a tweet object.

    Args:
        tweet: Raw tweet dictionary from JSONL

    Returns:
        Dictionary with only the required columns
    """
    author = tweet.get("author", {}) or {}

    return {
        "id": _to_str(tweet.get("id")),
        "text": _to_str(tweet.get("text")),
        "retweetCount": _to_int(tweet.get("retweetCount")),
        "replyCount": _to_int(tweet.get("replyCount")),
        "likeCount": _to_int(tweet.get("likeCount")),
        "quoteCount": _to_int(tweet.get("quoteCount")),
        "viewCount": _to_int(tweet.get("viewCount")),
        "createdAt": _to_str(tweet.get("createdAt")),
        "lang": _to_str(tweet.get("lang")),
        "bookmarkCount": _to_int(tweet.get("bookmarkCount")),
        "isReply": _to_bool(tweet.get("isReply")),
        "inReplyToId": _to_str(tweet.get("inReplyToId")),
        "conversationId": _to_str(tweet.get("conversationId")),
        "author.type": _to_str(author.get("type")),
        "author.userName": _to_str(author.get("userName")),
        "author.id": _to_str(author.get("id")),
        "author.name": _to_str(author.get("name")),
        "author.isBlueVerified": _to_bool(author.get("isBlueVerified")),
        "author.verifiedType": _to_str(author.get("verifiedType")),
        "author.followers": _to_int(author.get("followers")),
        "author.following": _to_int(author.get("following")),
        "author.location": _to_str(author.get("location")),
    }


def read_jsonl_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Read tweets from a JSONL file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of tweet dictionaries
    """
    tweets = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                tweet = json.loads(line)
                tweets.append(tweet)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num} in {file_path}: {e}")
    return tweets


def convert_to_parquet(tweets: List[Dict[str, Any]], output_file: Path, target_date: Optional[date] = None) -> int:
    """
    Convert tweets to Parquet format.

    Args:
        tweets: List of raw tweet dictionaries
        output_file: Path to output Parquet file
        target_date: Target date to filter tweets (extracted from filename)

    Returns:
        Number of tweets written
    """
    if not tweets:
        logger.warning(f"No tweets to convert for {output_file}")
        return 0

    # Extract required columns from each tweet
    extracted = [extract_tweet_data(tweet) for tweet in tweets]

    df = pd.DataFrame(extracted)
    
    initial_count = len(df)
    
    # Filter tweets by target date from filename

    def parse_created_at(created_at_str: str) -> date:
        try:
            return datetime.strptime(created_at_str, "%a %b %d %H:%M:%S +0000 %Y").date()
        except ValueError:
            return None
    
    df["_parsed_date"] = df["createdAt"].apply(parse_created_at)
    df = df[df["_parsed_date"] == target_date].copy()
    df = df.drop(columns=["_parsed_date"])
    
    # Drop duplicate tweet IDs within the same day
    df = df.drop_duplicates(subset=["id"], keep="first")
    logger.info(f"Removed {len(df) - initial_count}  tweet IDs")
    
    df.to_parquet(output_file, engine="fastparquet", compression="snappy", index=False)

    return len(df)


def convert_file(
    input_file: Path,
    output_dir: Path,
) -> int:
    """
    Convert a single JSONL file to Parquet.

    Args:
        input_file: Path to input JSONL file
        output_dir: Directory for output Parquet file

    Returns:
        Number of tweets converted
    """
    # Read JSONL
    tweets = read_jsonl_file(input_file)

    if not tweets:
        logger.warning(f"No tweets found in {input_file}")
        return 0

    # Extract date from filename (e.g., "tweets_2025-03-31.jsonl" -> "2025-03-31")
    target_date = None
    filename = input_file.stem  # Get filename without extension
    date_str = filename.split("_")[-1]  # Get last part after underscore
    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    logger.info(f"Extracted target date from filename: {target_date}")
    # Create output filename
    output_file = output_dir / input_file.name.replace(".jsonl", ".parquet")

    # Convert to Parquet
    count = convert_to_parquet(tweets, output_file, target_date)

    logger.info(f"Converted {count} tweets: {input_file.name} -> {output_file.name}")

    return count


def convert(
    input_file: Optional[str] = None,
    input_dir: str = DATA_DIR,
    output_dir: str = PARQUET_OUTPUT_DIR,
):
    """
    Convert JSONL files to Parquet format.

    Args:
        input_file: Optional specific JSONL file to convert
        input_dir: Directory containing JSONL files (default from config: DATA_DIR)
        output_dir: Directory for output Parquet files (default from config: PARQUET_OUTPUT_DIR)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if input_file:
        # Convert single file
        input_path = Path(input_file)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_file}")
            return

        count = convert_file(input_path, output_path)
        print(f"\nConverted {count} tweets from {input_file}")

    else:
        # Convert all JSONL files in input directory
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return

        jsonl_files = sorted(input_path.glob("*.jsonl"))

        if not jsonl_files:
            logger.warning(f"No JSONL files found in {input_dir}")
            return

        logger.info(f"Found {len(jsonl_files)} JSONL files to convert")

        total_tweets = 0
        for jsonl_file in tqdm(jsonl_files, desc="Converting files"):
            count = convert_file(jsonl_file, output_path)
            total_tweets += count

        print(f"\n=== Conversion Summary ===")
        print(f"Files converted: {len(jsonl_files)}")
        print(f"Total tweets: {total_tweets}")
        print(f"Output directory: {output_path}")


def convert_one_day(date: str):
    """
    Convert a single day of tweets to Parquet.
    tweets_YYYY-MM-DD.jsonl -> tweets_YYYY-MM-DD.parquet
    """
    json_files = Path(DATA_DIR).glob(f"tweets_{date}.jsonl")
    for json_file in json_files:
        count = convert_file(json_file, Path(PARQUET_OUTPUT_DIR))
        print(f"Converted {count} tweets: {json_file.name} ")


def main():
    """Main entry point using fire package."""
    fire.Fire({
        "convert": convert,
        "convert_one_day": convert_one_day,
    })


if __name__ == "__main__":
    main()

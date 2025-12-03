"""
Twitter API crawler using twitterapi.io

This script crawls tweets based on specific keywords and date ranges.
It supports test mode and formal crawl mode using the fire package.

Example usage:
    # Test mode
    python tweet_crawler.py test

    # Formal crawl mode
    python tweet_crawler.py crawl
"""

import requests
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import fire
import time
import random

# Import configuration
from config import (
    API_URL,
    API_KEY,
    TWEETS_PER_DATE,
    TWEETS_PER_PAGE,
    CALLS_PER_DATE,
    STATE_FILE,
    LOG_FILE,
    DATA_DIR,
    TEST_OUTPUT_FILE,
    KEYWORDS,
    START_DATE,
    END_DATE,
    TARGET_DAYS,
    LANGUAGE,
    QUERY_TYPE,
    REQUEST_DELAY,
)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2
REQUEST_TIMEOUT = 30

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def append_tweets(output_file: Path, tweets: List[Dict[str, Any]]):
    """Append tweets as JSON lines, one per line."""
    if not tweets:
        return
    with open(output_file, "a") as f:
        for tweet in tweets:
            f.write(json.dumps(tweet))
            f.write("\n")


def generate_dates() -> List[str]:
    """
    Generate dates from END_DATE to START_DATE in reverse order.
    Returns dates in format YYYY-MM-DD for days specified in TARGET_DAYS.
    """
    dates = []
    start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_date = datetime.strptime(END_DATE, "%Y-%m-%d")

    current = end_date
    while current >= start_date:
        if current.day in TARGET_DAYS:
            dates.append(current.strftime("%Y-%m-%d"))

        # Move to previous day
        current = current - timedelta(days=1)

    return dates


def build_query(
    keywords: List[str],
    date: str,
    language: str = LANGUAGE,
    max_id: Optional[str] = None,
    until_time_str: Optional[str] = None,
) -> str:
    """
    Build the query string for Twitter API.

    Args:
        keywords: List of keywords to search for
        date: Date in YYYY-MM-DD format
        language: Language code (default: en)
        max_id: Optional max_id to continue pagination beyond initial limit

    Returns:
        Query string formatted for Twitter API
    """
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    since_date = date_obj.strftime("%Y-%m-%d")
    until_date = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")

    # Create OR query for all keywords
    keyword_query = " OR ".join([f'"{kw}"' if " " in kw else kw for kw in keywords])

    # Add max_id if provided
    if max_id:
        query = f"({keyword_query}) lang:{language} since:{since_date} max_id:{max_id}"
    elif until_time_str:
        # until time str format: "%a %b %d %H:%M:%S +0000 %Y"
        # what should be in the query: since/until:2021-12-31_23:59:59_UTC
        until_str = datetime.strptime(
            until_time_str, "%a %b %d %H:%M:%S +0000 %Y"
        ).strftime("%Y-%m-%d_%H:%M:%S_UTC")
        since_str = datetime.strptime(since_date, "%Y-%m-%d").strftime(
            "%Y-%m-%d_%H:%M:%S_UTC"
        )
        query = f"({keyword_query}) lang:{language} since:{since_str} until:{until_str}"
    else:
        query = (
            f"({keyword_query}) lang:{language} since:{since_date} until:{until_date}"
        )

    return query


def load_state(state_file: str = STATE_FILE) -> Dict[str, Any]:
    """Load the state file to resume crawling."""
    state_path = Path(state_file)
    if state_path.exists():
        with open(state_path, "r") as f:
            return json.load(f)
    return {}


def save_state(state: Dict[str, Any], state_file: str = STATE_FILE):
    """Save the current state to file."""
    with open(state_file, "w") as f:
        json.dump(state, indent=2, fp=f)


def crawl_tweets_for_date(
    date: str,
    max_calls: int = None,
    test_mode: bool = False,
    start_cursor: str = "",
    start_calls: int = 0,
    output_file: Optional[Path] = None,
    state: Optional[Dict[str, Any]] = None,
    total_written: int = 0,
    start_max_id: Optional[str] = None,
    until_time_str: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Crawl tweets for a specific date.

    Args:
        date: Date in YYYY-MM-DD format
        max_calls: Maximum number of API calls (None for unlimited)
        test_mode: If True, only make 1 API call for testing
        start_cursor: Cursor to resume from
        start_calls: How many calls have already been made for this date
        output_file: If provided, append tweets to this file as JSONL per response
        state: Optional shared state dict to update/save per response
        total_written: Starting count of tweets already written for this date
        start_max_id: Optional max_id to resume from (for pagination beyond initial limit)
        until_time_str: Optional until time string to resume from

    Returns:
        Dictionary with crawl results
    """
    base_query = build_query(KEYWORDS, date)
    headers = {"X-API-Key": API_KEY}

    until_time_str = until_time_str or ""
    cursor = start_cursor or ""
    max_id = start_max_id
    previous_max_id = None
    previous_cursor = None
    calls_made = start_calls
    max_calls = 1 if test_mode else (max_calls or CALLS_PER_DATE)
    completed = False
    last_batch: List[Dict[str, Any]] = []
    max_id_retry_count = 0  # 跟踪 max_id 未推进的重试次数

    if calls_made >= max_calls:
        return {
            "date": date,
            "total_tweets": total_written,
            "calls_made": calls_made,
            "last_cursor": cursor,
            "tweets": last_batch,
            "finished": True,
        }

    logger.info(
        f"Starting crawl for {date} (max_calls: {max_calls}, resume_calls: {calls_made}, max_id: {max_id})"
    )

    with tqdm(
        total=max_calls, initial=calls_made, desc=f"Crawling {date}", disable=test_mode
    ) as pbar:
        while calls_made < max_calls:
            try:
                query = base_query

                params = {"queryType": QUERY_TYPE, "query": query}

                if cursor:
                    params["cursor"] = cursor
                elif max_id:
                    if max_id_retry_count >= 2:
                        query = build_query(
                            KEYWORDS,
                            date,
                            max_id=str(int(max_id) - (1000000000000 * max_id_retry_count)),
                        )
                    else:
                        query = build_query(KEYWORDS, date, max_id=max_id)
                    params["query"] = query
                # elif max_id and max_id_retry_count > 0:
                #     query = build_query(KEYWORDS, date, until_time_str=until_time_str)
                #     params["query"] = query

                # Make API request
                attempt = 0
                data = None
                while attempt < MAX_RETRIES:
                    try:
                        response = requests.get(
                            API_URL,
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
                                f"HTTP error ({status}) for {date}, retry {attempt}/{MAX_RETRIES}"
                            )
                            time.sleep(RETRY_DELAY)
                            continue
                        logger.error(f"HTTP error for {date}: {e}")
                        break
                    except (
                        requests.exceptions.RequestException,
                        json.JSONDecodeError,
                    ) as e:
                        attempt += 1
                        if attempt < MAX_RETRIES:
                            logger.warning(
                                f"Request/JSON error for {date}: {e} (retry {attempt}/{MAX_RETRIES})"
                            )
                            time.sleep(RETRY_DELAY)
                            continue
                        logger.error(f"Request/JSON error for {date}: {e}")
                        break

                if data is None:
                    logger.error(
                        f"Failed to fetch data for {date} after {MAX_RETRIES} retries, skipping"
                    )
                    break

                calls_made += 1
                pbar.update(1)

                # Check for errors in response
                if "error" in data:
                    logger.error(
                        f"API error for {date}: {data.get('message', 'Unknown error')}"
                    )
                    break

                # Extract tweets
                tweets = data.get("tweets", [])
                last_batch = tweets

                logger.info(
                    f"Call {calls_made}/{max_calls}: Retrieved {len(tweets)} tweets (total: {total_written})"
                )

                # Check if there are more pages
                has_next_page = data.get("has_next_page", False)
                next_cursor = data.get("next_cursor")
                new_until_time_str = data.get("createdAt")

                # Update max_id from the last tweet if we have tweets
                if tweets:
                    last_tweet_id = tweets[-1].get("id")
                    # check createdat
                    created_at = tweets[-1].get("createdAt")
                    created_at_date = datetime.strptime(
                        created_at, "%a %b %d %H:%M:%S +0000 %Y"
                    )

                    if created_at_date.date() == datetime.strptime(date, "%Y-%m-%d").date() and int(last_tweet_id) < int(max_id):
                        max_id = last_tweet_id
                        until_time_str = created_at
                        logger.info(f"Updated max_id for {date}: {max_id}")

                    else:
                        # Find the minimum valid tweet ID for the target date
                        min_valid_id = None
                        min_valid_created_at = None
                        for tweet in tweets:
                            if tweet.get("createdAt") and tweet.get("id"):
                                tweet_created_at = tweet.get("createdAt")
                                tweet_created_at_date = datetime.strptime(
                                    tweet_created_at, "%a %b %d %H:%M:%S +0000 %Y"
                                )
                                if (
                                    tweet_created_at_date.date()
                                    == datetime.strptime(date, "%Y-%m-%d").date()
                                ):
                                    tweet_id = tweet.get("id")
                                    if (not min_valid_id) or (
                                        int(tweet_id) < int(min_valid_id)
                                    ):
                                        min_valid_id = tweet_id
                                        min_valid_created_at = tweet_created_at

                        # Update max_id if we found valid tweets for the target date
                        if min_valid_id:
                            # If max_id is None (first call), initialize it
                            if max_id is None:
                                max_id = min_valid_id
                                until_time_str = min_valid_created_at
                                logger.info(f"Initialized max_id for {date}: {max_id}")
                            # If we found a smaller ID, update max_id (for pagination)
                            elif int(min_valid_id) < int(max_id):
                                max_id = min_valid_id
                                until_time_str = min_valid_created_at
                                logger.info(f"Max_id {date}: {max_id}, Total {total_written}")
                            else:
                                # No smaller ID found, might have reached the end
                                logger.warning(
                                    f"No smaller ID found for {date} (min_valid_id: {min_valid_id}, max_id: {max_id})"
                                )
                                next_cursor = None
                        else:
                            # No valid tweets for target date in this batch
                            logger.warning(
                                f"No valid tweets for target date {date} in this batch"
                            )
                            next_cursor = None
                            # Don't set next_cursor to None here, let the API's has_next_page decide

                # Update cursor for next iteration (if using cursor-based pagination)
                if next_cursor and (next_cursor != cursor):
                    cursor = next_cursor
                else:
                    # No more pages, clear cursor
                    cursor = None
                
                # Check if max_id advanced (only if both are not None)
                if (
                    max_id is not None
                    and previous_max_id is not None
                    and max_id == previous_max_id
                ):
                    max_id_retry_count += 1

                    logger.warning(
                        f"max_id did not advance for {date} after {max_id_retry_count}/5 retries (max_id: {max_id})"
                    )
                    if max_id_retry_count >= 5:
                        completed = True
                else:
                    max_id_retry_count = 0

                previous_max_id = max_id

                finished_now = calls_made >= max_calls

                append_tweets(output_file, tweets)

                total_written += len(tweets)

                if state is not None:
                    state[date] = {
                        "finished": finished_now,
                        "total_tweets": total_written,
                        "calls_made": calls_made,
                        "current_cursor": cursor,
                        "max_id": max_id,
                        "until_time_str": until_time_str,
                    }
                    save_state(state)

                if completed:
                    break

                # Small delay to avoid rate limiting
                time.sleep(REQUEST_DELAY)

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error for {date}: {e}")
                break
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error for {date}: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error for {date}: {e}")
                break

    return {
        "date": date,
        "total_tweets": total_written,
        "calls_made": calls_made,
        "last_cursor": cursor,
        "max_id": max_id,
        "tweets": last_batch,
        "finished": completed or calls_made >= max_calls,
        "total_written": total_written,
        "until_time_str": until_time_str,
    }


def test():
    """
    Test mode: Crawl 1 request each for 2024-03-01 and 2025-03-20.
    Save results to test.jsonl (one tweet per line).
    """
    logger.info("=== Starting TEST mode ===")

    test_file = Path(TEST_OUTPUT_FILE)
    test_file.parent.mkdir(parents=True, exist_ok=True)

    test_dates = ["2024-03-01", "2025-03-20"]
    test_results = {}

    for date in test_dates:
        logger.info(f"Testing date: {date}")
        result = crawl_tweets_for_date(
            date, test_mode=True, output_file=test_file, total_written=0
        )
        test_results[date] = {
            "total_tweets": result["total_tweets"],
            "calls_made": result["calls_made"],
            "query": build_query(KEYWORDS, date),
            "tweets": result["tweets"],
        }
        logger.info(
            f"Test for {date} completed: {result['total_tweets']} tweets retrieved"
        )

    logger.info(f"Test tweets appended to {test_file}")
    logger.info("=== TEST mode completed ===")

    # Print summary
    print("\n=== Test Summary ===")
    for date, result in test_results.items():
        print(
            f"{date}: {result['total_tweets']} tweets, {result['calls_made']} API calls"
        )
        print(f"  Query: {result['query']}")


def crawl():
    """
    Formal crawl mode: Crawl all dates with full configuration.
    Saves each date to a JSONL file (one tweet per line) and persists after each response.
    Maintains state file for resuming interrupted crawls.
    """
    logger.info("=== Starting FORMAL CRAWL mode ===")

    # Generate all dates
    dates = generate_dates()
    logger.info(f"Total dates to crawl: {len(dates)}")
    logger.info(f"Date range: {dates[-1]} to {dates[0]}")

    # Load existing state
    state = load_state()
    logger.info(f"Loaded state with {len(state)} entries")

    # Create data directory
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(exist_ok=True)

    # Overall progress bar
    completed_dates = sum(1 for d in dates if state.get(d, {}).get("finished", False))

    with tqdm(
        total=len(dates), initial=completed_dates, desc="Overall Progress"
    ) as overall_pbar:
        for date in dates:
            # Skip if already finished
            if state.get(date, {}).get("finished", False):
                logger.info(f"Skipping {date} (already completed)")
                continue

            # Get resume cursor and max_id if exists
            resume_cursor = state.get(date, {}).get("current_cursor", "")
            resume_calls = state.get(date, {}).get("calls_made", 0)
            resume_max_id = state.get(date, {}).get("max_id")
            resume_until_time_str = state.get(date, {}).get("until_time_str")

            logger.info(
                f"Starting crawl for {date} (resume from call {resume_calls}, max_id: {resume_max_id})"
            )

            # Crawl tweets for this date
            output_file = data_dir / f"tweets_{date}.jsonl"
            total_written = state.get(date, {}).get("total_tweets", 0)

            result = crawl_tweets_for_date(
                date,
                start_cursor=resume_cursor,
                start_calls=resume_calls,
                output_file=output_file,
                state=state,
                total_written=total_written,
                start_max_id=resume_max_id,
                until_time_str=resume_until_time_str,
            )

            # Ensure state is saved even if no batches were processed
            if (
                date not in state
                or state[date].get("calls_made") != result["calls_made"]
            ):
                state[date] = {
                    "finished": result["finished"],
                    "total_tweets": result["total_written"],
                    "calls_made": result["calls_made"],
                    "current_cursor": result["last_cursor"],
                    "max_id": result.get("max_id"),
                    "until_time_str": result.get("until_time_str"),
                }
                save_state(state)

            if result["finished"]:
                overall_pbar.update(1)

    logger.info("=== FORMAL CRAWL mode completed ===")

    # Print final summary
    print("\n=== Crawl Summary ===")
    total_tweets = sum(s.get("total_tweets", 0) for s in state.values())
    total_calls = sum(s.get("calls_made", 0) for s in state.values())
    print(f"Total dates processed: {len(state)}")
    print(f"Total tweets collected: {total_tweets}")
    print(f"Total API calls made: {total_calls}")


def crawl_twenty_rounds():
    for i in range(100):
        crawl()
        sleep_time = random.randint(10, 60)
        time.sleep(sleep_time)
        logger.info(f"Crawled {i+1} rounds, sleeping for {sleep_time} seconds")

def count_total_tweets():
    state_data = load_state()
    total_tweets = 0

    for date, data in state_data.items():
        if date.endswith("01") or date.endswith("10") or date.endswith("20"):
            total_tweets += data.get("total_tweets", 0)

    print(f"Total tweets collected: {total_tweets}")


if __name__ == "__main__":
    fire.Fire({"test": test, "crawl": crawl, "crawl20": crawl_twenty_rounds, "count": count_total_tweets})

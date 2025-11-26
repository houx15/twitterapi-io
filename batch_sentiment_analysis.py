"""
Batch Sentiment Analysis for Twitter Data using OpenAI Batch API

This script uses OpenAI's batch API to analyze tweet sentiment in batches.
It samples tweets from dates on 01, 10, 20 of each month and processes them
using the /v1/responses endpoint.

Usage:
    # Submit a batch
    python batch_sentiment_analysis.py --content sample --action submit

    # Retrieve batch results
    python batch_sentiment_analysis.py --content sample --action retrieve

    # Analyze results and generate summary
    python batch_sentiment_analysis.py --content sample --action analyze

    # Process all batches
    python batch_sentiment_analysis.py --content all --action submit
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import requests
import fire
import jsonlines

from config import (
    PARQUET_OUTPUT_DIR,
    OPENAI_API_KEY,
    SENTIMENT_OUTPUT_DIR,
    SENTIMENT_SAMPLE_SIZE,
    TARGET_DAYS,
    BATCH_MAX_LINES,
    BATCH_PROMPT_ID,
    BATCH_PROMPT_VERSION,
    BATCH_MODEL,
    BATCH_LIST_FILE,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Derived paths
BATCH_BASE_DIR = Path(SENTIMENT_OUTPUT_DIR) / "batches"
CACHE_FILE = Path(SENTIMENT_OUTPUT_DIR) / "processed_ids.txt"


def load_processed_ids() -> Set[str]:
    """Load set of already processed tweet IDs from cache file."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r") as f:
                return set(f.read().splitlines())
        except Exception as e:
            logger.warning(f"Failed to load cache file: {e}")
    return set()


def save_processed_id(new_id: str):
    """Save processed tweet IDs to cache file."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "a") as f:
        f.write(new_id + "\n")


def load_parquet_files_filtered_by_date(
    input_dir: str = PARQUET_OUTPUT_DIR,
) -> pd.DataFrame:
    """
    Load tweets from parquet files, filtering by filename dates matching TARGET_DAYS.

    Args:
        input_dir: Directory containing parquet files

    Returns:
        DataFrame containing filtered tweets
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    parquet_files = sorted(input_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    logger.info(f"Found {len(parquet_files)} parquet files")

    dfs = []
    for pf in tqdm(parquet_files, desc="Loading parquet files"):
        try:
            # Extract date from filename (format: tweets_YYYY-MM-DD.parquet)
            filename = pf.stem
            if "tweets_" in filename:
                date_str = filename.replace("tweets_", "")
                try:
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                    # Filter by target days from config
                    if date_obj.day in TARGET_DAYS:
                        df = pd.read_parquet(pf, columns=["id", "text"])
                        dfs.append(df)
                        logger.debug(
                            f"Loaded {len(df)} tweets from {pf.name} (date: {date_str}, day: {date_obj.day})"
                        )
                except ValueError:
                    logger.warning(f"Failed to parse date from filename: {filename}")
            else:
                logger.debug(f"Skipping {pf.name} (filename format not recognized)")
        except Exception as e:
            logger.warning(f"Failed to load {pf}: {e}")

    if not dfs:
        raise ValueError(f"No tweets found matching date filter (days: {TARGET_DAYS})")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined)} tweets from dates matching {TARGET_DAYS}")

    return combined


def sample_tweets(
    df: pd.DataFrame,
    sample_size: Optional[int] = None,
    seed: int = 42,
    exclude_ids: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Sample tweets from DataFrame, excluding already processed IDs.
    If sample_size is None, return all tweets.

    Args:
        df: DataFrame containing tweets
        sample_size: Number of tweets to sample (None for all)
        seed: Random seed for reproducibility
        exclude_ids: Set of tweet IDs to exclude

    Returns:
        Sampled DataFrame
    """
    if exclude_ids:
        df = df[~df["id"].isin(exclude_ids)].copy()
        logger.info(f"Excluded {len(exclude_ids)} already processed tweets")
    
    # 不能重复
    df = df.drop_duplicates(subset=["id"])

    if len(df) == 0:
        raise ValueError("No tweets available after filtering")

    if sample_size is None:
        # Return all tweets
        logger.info(f"Using all {len(df)} tweets (no sampling)")
        return df

    if len(df) <= sample_size:
        logger.warning(
            f"DataFrame has {len(df)} rows, less than sample size {sample_size}"
        )
        return df

    sampled = df.sample(n=sample_size, random_state=seed)
    logger.info(f"Sampled {len(sampled)} tweets from {len(df)} total")

    return sampled


def create_batch_request(tweet_id: str, tweet_text: str) -> Dict[str, Any]:
    """
    Create a batch request for a single tweet.

    Args:
        tweet_id: Twitter ID
        tweet_text: Tweet text content

    Returns:
        Batch request dictionary
    """
    timestamp = str(int(time.time()))
    return {
        "custom_id": f"{timestamp}-{tweet_id}",
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": BATCH_MODEL,
            "reasoning": {"effort": "medium"},
            "text": {"format": {"type": "json_object"}},
            "prompt": {"id": BATCH_PROMPT_ID, "version": BATCH_PROMPT_VERSION},
            "input": f"Please output the results in JSON format. \nTweet text: {tweet_text}"
        },
    }


def split_into_batches(
    requests: List[Dict[str, Any]], max_lines: int = BATCH_MAX_LINES
) -> List[List[Dict[str, Any]]]:
    """
    Split requests into batches of max_lines size.

    Args:
        requests: List of batch requests
        max_lines: Maximum lines per batch

    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(requests), max_lines):
        batches.append(requests[i : i + max_lines])
    return batches


class BatchManager:
    """Manager for batch operations and state."""

    def __init__(self):
        """Initialize batch manager."""
        self.batch_list_path = Path(BATCH_LIST_FILE)
        self._batch_list: Optional[Dict[int, Dict[str, Any]]] = None

    def load(self) -> Dict[int, Dict[str, Any]]:
        """Load batch list from file, return as dict keyed by index."""
        if self._batch_list is not None:
            return self._batch_list

        if self.batch_list_path.exists():
            try:
                with open(self.batch_list_path, "r") as f:
                    batch_list = json.load(f)
                    # Convert list to dict keyed by index
                    self._batch_list = {batch["index"]: batch for batch in batch_list}
                    return self._batch_list
            except Exception as e:
                logger.warning(f"Failed to load batch list: {e}")

        self._batch_list = {}
        return self._batch_list

    def save(self):
        """Save batch list to file."""
        if self._batch_list is None:
            self.load()

        self.batch_list_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert dict to list for JSON storage
        batch_list = sorted(self._batch_list.values(), key=lambda x: x.get("index", 0))
        with open(self.batch_list_path, "w") as f:
            json.dump(batch_list, f, indent=2)
        logger.debug(f"Saved {len(batch_list)} batches to batch list")

    def get_next_index(self) -> int:
        """Get next available batch index."""
        batch_dict = self.load()
        if not batch_dict:
            return 0
        max_index = max(batch_dict.keys(), default=-1)
        return max_index + 1

    def get_batch(self, index: int) -> Optional[Dict[str, Any]]:
        """Get batch by index."""
        batch_dict = self.load()
        return batch_dict.get(index)

    def add_batch(self, batch_info: Dict[str, Any]):
        """Add or update a batch in the list."""
        if self._batch_list is None:
            self.load()

        index = batch_info["index"]
        # Merge with existing batch info if present
        if index in self._batch_list:
            self._batch_list[index].update(batch_info)
        else:
            self._batch_list[index] = batch_info

    def update_batch(self, index: int, updates: Dict[str, Any]):
        """Update batch information."""
        if self._batch_list is None:
            self.load()

        if index in self._batch_list:
            self._batch_list[index].update(updates)
        else:
            logger.warning(f"Batch {index} not found in batch list")

    def get_batches_by_mode(self, content_mode: str) -> List[Dict[str, Any]]:
        """Get batches filtered by content mode."""
        batch_dict = self.load()
        return [
            batch
            for batch in batch_dict.values()
            if batch.get("content_mode") == content_mode
        ]

    def get_all_batches(self) -> List[Dict[str, Any]]:
        """Get all batches as list."""
        batch_dict = self.load()
        return sorted(batch_dict.values(), key=lambda x: x.get("index", 0))


def get_batch_dir(batch_index: int) -> Path:
    """Get directory path for a batch."""
    batch_dir = BATCH_BASE_DIR / f"batch_{batch_index:04d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    return batch_dir


def submit_batch(
    content: str = "sample",
    sample_size: Optional[int] = None,
    seed: int = 42,
):
    """
    Submit batch requests to OpenAI API.

    Args:
        content: "sample" to sample 1000 tweets, "all" to process all tweets
        sample_size: Number of tweets to sample (only used if content="sample", None uses default)
        seed: Random seed for reproducibility
    """
    logger.info("=== Starting Batch Submission ===")

    # Initialize OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)

    batch_manager = BatchManager()

    # Load processed IDs (tweets already placed into a batch file)
    processed_ids = load_processed_ids()
    logger.info(f"Loaded {len(processed_ids)} processed IDs")

    # Load and filter tweets (only once)
    df = load_parquet_files_filtered_by_date()

    created_batches: List[int] = []

    def build_requests(df_subset: pd.DataFrame) -> List[Dict[str, Any]]:
        requests: List[Dict[str, Any]] = []
        for _, row in tqdm(
            df_subset.iterrows(), total=len(df_subset), desc="Creating requests"
        ):
            tweet_id = row.get("id", "")
            tweet_text = row.get("text", "") or ""
            if tweet_id and tweet_text:
                requests.append(create_batch_request(str(tweet_id), tweet_text))
                save_processed_id(str(tweet_id))
        return requests

    def write_requests_chunk(requests: List[Dict[str, Any]], mode: str) -> int:
        batch_index = batch_manager.get_next_index()
        batch_dir = get_batch_dir(batch_index)
        requests_file = batch_dir / "requests.jsonl"
        with jsonlines.open(requests_file, mode="w") as writer:
            for req in requests:
                writer.write(req)
        batch_manager.add_batch(
            {
                "index": batch_index,
                "status": "created",
                "request_count": len(requests),
                "content_mode": mode,
                "requests_file": str(requests_file),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        return batch_index

    # Step 1: creation (requests file only)
    if content == "sample":
        if batch_manager.get_batches_by_mode("sample"):
            logger.info("Sample batch already exists; skipping new sample creation")
        else:
            actual_sample_size = (
                sample_size if sample_size is not None else SENTIMENT_SAMPLE_SIZE
            )
            sampled_df = sample_tweets(
                df, sample_size=actual_sample_size, seed=seed, exclude_ids=processed_ids
            )
            logger.info("Creating sample batch requests...")
            batch_requests = build_requests(sampled_df)
            created_batches.append(write_requests_chunk(batch_requests, content))
    else:
        sampled_df = sample_tweets(
            df, sample_size=None, seed=seed, exclude_ids=processed_ids
        )
        logger.info("Creating all-mode batch requests...")
        batch_requests = build_requests(sampled_df)
        for chunk in split_into_batches(batch_requests):
            created_batches.append(write_requests_chunk(chunk, content))

    if created_batches:
        logger.info(f"Created batches: {created_batches}")
    
    batch_manager.save()

    target_batches = batch_manager.get_batches_by_mode(content)

    # Step 2: upload files for batches needing a file_id
    for batch_info in target_batches:
        batch_index = batch_info.get("index")
        status = batch_info.get("status")
        file_id = batch_info.get("file_id")
        requests_file = batch_info.get("requests_file")
        if status in {"created", "upload_failed"} or not file_id:
            if not requests_file or not Path(requests_file).exists():
                print(f"Batch {batch_index} missing requests file; cannot upload")
                continue
            try:
                with open(requests_file, "rb") as f:
                    uploaded_file = client.files.create(file=f, purpose="batch")
                file_id = uploaded_file.id
                logger.info(f"Uploaded file for batch {batch_index}: {file_id}")
                batch_manager.update_batch(
                    batch_index,
                    {
                        "file_id": file_id,
                        "status": "uploaded",
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
            except Exception as e:
                logger.error(f"Failed to upload file for batch {batch_index}: {e}")
                pass
    
    batch_manager.save()

    # Step 3: create batches for entries with a file_id but no batch_id
    for batch_info in target_batches:
        batch_index = batch_info.get("index")
        file_id = batch_info.get("file_id")
        batch_id = batch_info.get("batch_id")
        status = batch_info.get("status")
        if file_id and (not batch_id or status == "uploaded"):
            try:
                batch = client.batches.create(
                    input_file_id=file_id,
                    endpoint="/v1/responses",
                    completion_window="24h",

                )
                batch_id = batch.id
                logger.info(f"Batch {batch_index} submitted with ID: {batch_id}")
                batch_manager.update_batch(
                    batch_index,
                    {
                        "batch_id": batch_id,
                        "status": "submitted",
                        "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
            except Exception as e:
                logger.error(f"Failed to create batch {batch_index}: {e}")
                pass

    # Save updated batch list
    batch_manager.save()

    logger.info("=== Batch Submission Complete ===")


def retrieve_batch(content: str = "sample"):
    """
    Retrieve batch results from OpenAI API.

    Args:
        content: Ignored; retrieval checks all batches
    """
    logger.info("=== Starting Batch Retrieval ===")

    # Initialize OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Initialize batch manager
    batch_manager = BatchManager()

    batches_to_process = [
        b
        for b in batch_manager.get_all_batches()
        if (b.get("batch_id") and b.get("status") == "submitted")
    ]

    if not batches_to_process:
        logger.warning("No batches with batch_id found to retrieve")
        return

    logger.info(f"Found {len(batches_to_process)} batches to retrieve")

    retrieved_batches = []
    running_batches = []

    for batch_info in tqdm(batches_to_process, desc="Retrieving batches"):
        batch_index = batch_info.get("index")
        batch_id = batch_info.get("batch_id")

        try:
            batch = client.batches.retrieve(batch_id)
            status = batch.status
            logger.info(f"Batch {batch_index}: status={status}")

            # updates = {
            #     "status": status,
            #     "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            # }
            # if hasattr(batch, "request_counts"):
            #     updates["request_count"] = batch.request_counts.get("total", 0)
            # batch_manager.update_batch(batch_index, updates)

            if status == "completed":
                batch_dir = get_batch_dir(batch_index)
                results_file = batch_dir / "results.jsonl"

                # if not results_file.exists():
                # logger.info(f"Downloading results for batch {batch_index}...")
                output_file_id = batch.output_file_id
                # if output_file_id:
                output_content = client.files.content(output_file_id)
                with open(results_file, "wb") as f:
                    f.write(output_content.read())
                logger.info(f"Saved results to {results_file}")
                batch_manager.update_batch(
                    batch_index,
                    {
                        "status": "retrieved",
                        "results_file": str(results_file),
                    },
                )
                retrieved_batches.append(batch_index)
            elif status == "expired":
                # Resubmit using existing file_id
                file_id = batch_info.get("file_id")
                logger.info(
                    f"Resubmitting expired batch {batch_index} with file_id {file_id}"
                )
                new_batch = client.batches.create(input_file_id=file_id)
                batch_manager.update_batch(
                    batch_index,
                    {
                        "batch_id": new_batch.id,
                        "status": "submitted",
                        "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                )
                running_batches.append(batch_index)
            else:
                running_batches.append(batch_index)

        except Exception as e:
            logger.error(f"Failed to retrieve batch {batch_index}: {e}")
            pass

    batch_manager.save()

    print("\n=== Batch Retrieval Report ===")
    if retrieved_batches:
        print(
            f"Retrieved results for batches: {', '.join(str(b) for b in sorted(retrieved_batches))}"
        )
    else:
        print("No batches completed in this run.")
    if running_batches:
        print(
            f"Batches still running/submitted: {', '.join(str(b) for b in sorted(set(running_batches)))}"
        )

    logger.info("=== Batch Retrieval Complete ===")


def analyze_batch_results(content: str = "sample"):
    """
    Analyze batch results and generate summary.

    Args:
        content: "sample" to process sample batches, "all" to process all batches
    """
    logger.info("=== Starting Batch Analysis ===")

    # Initialize batch manager
    batch_manager = BatchManager()

    # Get batches to process
    batches_to_process = batch_manager.get_batches_by_mode(content)

    if not batches_to_process:
        logger.warning(f"No batches found for content mode: {content}")
        return

    logger.info(f"Found {len(batches_to_process)} batches to analyze")

    all_results = set()

    for batch_info in tqdm(batches_to_process, desc="Analyzing batches"):
        batch_index = batch_info.get("index")

        if batch_info.get("status") != "retrieved":
            continue

        # Get results file path from batch info or construct it
        results_file_path = batch_info.get("results_file")
        if results_file_path:
            results_file = Path(results_file_path)
        else:
            batch_dir = get_batch_dir(batch_index)
            results_file = batch_dir / "results.jsonl"

        if not results_file.exists():
            logger.warning(
                f"Results file not found for batch {batch_index}: {results_file}"
            )
            continue

        # Read results using jsonlines
        with jsonlines.open(results_file) as reader:
            for result in reader:
                custom_id = result.get("custom_id")
                opinion = None
                outputs = result["response"]["body"]["output"]
                for output in outputs:
                    if output["type"] == "message":
                        opinion_json = output["content"][0]["text"]
                        try:
                            opinion = json.loads(opinion_json).get("opinion")
                        except Exception as e:
                            logger.error(f"Failed to parse opinion JSON: {e}, {opinion_json}")
                        if opinion is not None:
                            break
                all_results.add((custom_id, opinion))

    all_results = list(all_results)
    all_results_df = pd.DataFrame(all_results, columns=["custom_id", "opinion"])
    
    opinion_counts = all_results_df["opinion"].value_counts()
    opinion_percentages = opinion_counts / opinion_counts.sum() * 100

    print(f"Opinion counts: \n{opinion_counts}")
    print(f"Opinion percentages: \n{opinion_percentages}")
    
    all_results_df.to_parquet(Path(SENTIMENT_OUTPUT_DIR) / "batch_results.parquet")
    print(f"\nResults saved to: {SENTIMENT_OUTPUT_DIR} / 'batch_results.parquet'")

    logger.info("=== Batch Analysis Complete ===")


class BatchSentimentCLI:
    """CLI class for batch sentiment analysis."""

    def __init__(self, content: str = "sample"):
        """
        Initialize CLI.

        Args:
            content: Content to process ("sample" or "all")
        """
        self.content = content

    def submit(self, sample_size: Optional[int] = None, seed: int = 42):
        """Submit batch requests."""
        submit_batch(content=self.content, sample_size=sample_size, seed=seed)

    def retrieve(self):
        """Retrieve batch results."""
        retrieve_batch(content=self.content)

    def analyze(self):
        """Analyze batch results."""
        analyze_batch_results(content=self.content)


def main():
    """Main entry point using fire package."""
    fire.Fire(BatchSentimentCLI)


if __name__ == "__main__":
    main()

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
CACHE_FILE = Path(SENTIMENT_OUTPUT_DIR) / "processed_ids.json"


def load_processed_ids() -> Set[str]:
    """Load set of already processed tweet IDs from cache file."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)
                return set(data.get("processed_ids", []))
        except Exception as e:
            logger.warning(f"Failed to load cache file: {e}")
    return set()


def save_processed_ids(processed_ids: Set[str]):
    """Save processed tweet IDs to cache file."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {"processed_ids": list(processed_ids), "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(processed_ids)} processed IDs to cache")


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
                        logger.debug(f"Loaded {len(df)} tweets from {pf.name} (date: {date_str}, day: {date_obj.day})")
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
    return {
        "custom_id": str(tweet_id),
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": BATCH_MODEL,
            "reasoning": {"effort": "medium"},
            "prompt": {
                "id": BATCH_PROMPT_ID,
                "version": BATCH_PROMPT_VERSION,
                "variables": {
                    "text": tweet_text
                }
            }
        }
    }


def split_into_batches(requests: List[Dict[str, Any]], max_lines: int = BATCH_MAX_LINES) -> List[List[Dict[str, Any]]]:
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
        batches.append(requests[i:i + max_lines])
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
            batch for batch in batch_dict.values()
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
    
    # Load processed IDs
    processed_ids = load_processed_ids()
    logger.info(f"Loaded {len(processed_ids)} processed IDs from cache")
    
    # Load and filter tweets
    df = load_parquet_files_filtered_by_date()
    
    # Sample tweets based on content mode
    if content == "sample":
        # Sample mode: sample 1000 tweets (or specified sample_size)
        actual_sample_size = sample_size if sample_size is not None else SENTIMENT_SAMPLE_SIZE
        sampled_df = sample_tweets(df, sample_size=actual_sample_size, seed=seed, exclude_ids=processed_ids)
    else:
        # All mode: use all tweets
        sampled_df = sample_tweets(df, sample_size=None, seed=seed, exclude_ids=processed_ids)
    
    if len(sampled_df) == 0:
        logger.warning("No tweets to process after filtering")
        return
    
    # Create batch requests
    logger.info("Creating batch requests...")
    batch_requests = []
    for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Creating requests"):
        tweet_id = row.get("id", "")
        tweet_text = row.get("text", "") or ""
        if tweet_id and tweet_text:
            batch_requests.append(create_batch_request(str(tweet_id), tweet_text))
    
    # Split into batches
    batches = split_into_batches(batch_requests)
    logger.info(f"Split into {len(batches)} batches")
    
    # Initialize batch manager
    batch_manager = BatchManager()
    
    # Submit each batch
    for batch_reqs in batches:
        # Get next available batch index
        batch_index = batch_manager.get_next_index()
        batch_dir = get_batch_dir(batch_index)
        
        logger.info(f"Submitting batch {batch_index} ({len(batch_reqs)} requests)...")
        
        # Save requests to file
        requests_file = batch_dir / "requests.jsonl"
        
        # Check if batch already exists (e.g., from previous interrupted run)
        existing_batch = batch_manager.get_batch(batch_index)
        file_id = None
        batch_id = None
        
        if existing_batch:
            # Batch already exists, check if we can reuse file_id and batch_id
            file_id = existing_batch.get("file_id")
            batch_id = existing_batch.get("batch_id")
            if file_id:
                logger.info(f"Found existing batch {batch_index} with file_id {file_id}")
            if batch_id:
                logger.info(f"Found existing batch {batch_index} with batch_id {batch_id}")
        else:
            # New batch, write requests file
            with jsonlines.open(requests_file, mode="w") as writer:
                for req in batch_reqs:
                    writer.write(req)
        
        # Upload file if not already uploaded
        if not file_id:
            try:
                with open(requests_file, "rb") as f:
                    uploaded_file = client.files.create(
                        file=f,
                        purpose="batch"
                    )
                file_id = uploaded_file.id
                logger.info(f"Uploaded file {file_id} for batch {batch_index}")
            except Exception as e:
                logger.error(f"Failed to upload file for batch {batch_index}: {e}")
                batch_manager.add_batch({
                    "index": batch_index,
                    "batch_id": None,
                    "status": "failed",
                    "request_count": len(batch_reqs),
                    "error": f"File upload failed: {str(e)}",
                    "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "content_mode": content,
                    "requests_file": str(requests_file),
                })
                batch_manager.save()
                continue
        
        # Create batch if not already created
        if not batch_id:
            try:
                batch = client.batches.create(
                    input_file_id=file_id,
                )
                batch_id = batch.id
                logger.info(f"Batch {batch_index} submitted with ID: {batch_id}")
            except Exception as e:
                logger.error(f"Failed to create batch {batch_index}: {e}")
                batch_manager.add_batch({
                    "index": batch_index,
                    "batch_id": None,
                    "status": "failed",
                    "request_count": len(batch_reqs),
                    "error": str(e),
                    "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "content_mode": content,
                    "file_id": file_id,
                    "requests_file": str(requests_file),
                })
                batch_manager.save()
                continue
        
        # Update batch list with all information
        batch_manager.add_batch({
            "index": batch_index,
            "batch_id": batch_id,
            "status": "submitted",
            "request_count": len(batch_reqs),
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "content_mode": content,
            "file_id": file_id,
            "requests_file": str(requests_file),
        })
    
    # Save updated batch list
    batch_manager.save()
    
    logger.info("=== Batch Submission Complete ===")


def retrieve_batch(content: str = "sample"):
    """
    Retrieve batch results from OpenAI API.
    
    Args:
        content: "sample" to process sample batches, "all" to process all batches
    """
    logger.info("=== Starting Batch Retrieval ===")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Initialize batch manager
    batch_manager = BatchManager()
    
    # Get batches to process
    if content == "sample":
        batches_to_process = batch_manager.get_batches_by_mode("sample")
    else:
        batches_to_process = batch_manager.get_all_batches()
    
    if not batches_to_process:
        logger.warning(f"No batches found for content mode: {content}")
        return
    
    logger.info(f"Found {len(batches_to_process)} batches to retrieve")
    
    for batch_info in tqdm(batches_to_process, desc="Retrieving batches"):
        batch_index = batch_info.get("index")
        batch_id = batch_info.get("batch_id")
        
        if not batch_id:
            logger.warning(f"Skipping batch {batch_index} (no batch_id)")
            continue
        
        # Skip if already completed (only for sample mode to avoid re-checking)
        if batch_info.get("status") == "completed" and content == "sample":
            logger.debug(f"Skipping batch {batch_index} (already completed)")
            continue
        
        try:
            # Retrieve batch
            batch = client.batches.retrieve(batch_id)
            
            logger.info(f"Batch {batch_index}: status={batch.status}")
            
            # Update batch list
            updates = {
                "status": batch.status,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            
            # Add request counts if available
            if hasattr(batch, "request_counts"):
                updates["request_count"] = batch.request_counts.get("total", 0)
            
            batch_manager.update_batch(batch_index, updates)
            
            # If completed, download results
            if batch.status == "completed":
                batch_dir = get_batch_dir(batch_index)
                results_file = batch_dir / "results.jsonl"
                
                # Only download if not already downloaded
                if not results_file.exists():
                    logger.info(f"Downloading results for batch {batch_index}...")
                    
                    # Download output file
                    output_file_id = batch.output_file_id
                    if output_file_id:
                        output_content = client.files.content(output_file_id)
                        with open(results_file, "wb") as f:
                            f.write(output_content.read())
                        logger.info(f"Saved results to {results_file}")
                        batch_manager.update_batch(batch_index, {
                            "results_file": str(results_file),
                        })
                else:
                    logger.debug(f"Results file already exists for batch {batch_index}")
            
        except Exception as e:
            logger.error(f"Failed to retrieve batch {batch_index}: {e}")
            batch_manager.update_batch(batch_index, {
                "status": "error",
                "error": str(e),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
    
    # Save updated batch list
    batch_manager.save()
    
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
    if content == "sample":
        batches_to_process = batch_manager.get_batches_by_mode("sample")
    else:
        batches_to_process = batch_manager.get_all_batches()
    
    if not batches_to_process:
        logger.warning(f"No batches found for content mode: {content}")
        return
    
    logger.info(f"Found {len(batches_to_process)} batches to analyze")
    
    all_results = []
    all_usage = {
        "input_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }
    processed_ids = load_processed_ids()
    
    for batch_info in tqdm(batches_to_process, desc="Analyzing batches"):
        batch_index = batch_info.get("index")
        
        if batch_info.get("status") != "completed":
            logger.warning(f"Skipping batch {batch_index} (status: {batch_info.get('status')})")
            continue
        
        # Get results file path from batch info or construct it
        results_file_path = batch_info.get("results_file")
        if results_file_path:
            results_file = Path(results_file_path)
        else:
            batch_dir = get_batch_dir(batch_index)
            results_file = batch_dir / "results.jsonl"
        
        if not results_file.exists():
            logger.warning(f"Results file not found for batch {batch_index}: {results_file}")
            continue
        
        # Read results using jsonlines
        try:
            with jsonlines.open(results_file) as reader:
                for result in reader:
                    all_results.append(result)
                    
                    # Extract usage - check if response is successful
                    response = result.get("response", {})
                    status_code = response.get("status_code", 200)
                    
                    if status_code == 200 and "body" in response:
                        # Extract usage from response body if available
                        if "usage" in response:
                            usage = response["usage"]
                            all_usage["input_tokens"] += usage.get("input_tokens", 0)
                            all_usage["cached_tokens"] += usage.get("input_tokens_details", {}).get("cached_tokens", 0)
                            all_usage["output_tokens"] += usage.get("output_tokens", 0)
                            all_usage["reasoning_tokens"] += usage.get("output_tokens_details", {}).get("reasoning_tokens", 0)
                            all_usage["total_tokens"] += usage.get("total_tokens", 0)
                    
                    # Track processed ID (even if failed)
                    custom_id = result.get("custom_id")
                    if custom_id:
                        processed_ids.add(str(custom_id))
        except Exception as e:
            logger.warning(f"Failed to read results file for batch {batch_index}: {e}")
    
    # Save processed IDs
    save_processed_ids(processed_ids)
    
    # Parse opinions
    opinions = []
    for result in all_results:
        custom_id = result.get("custom_id")
        response = result.get("response", {})
        status_code = response.get("status_code", 200)
        
        # Extract opinion from response
        # OpenAI batch results format: response.body contains the actual response
        opinion = None
        error_message = None
        
        if status_code == 200 and "body" in response:
            try:
                # body might be a string or already parsed
                body = response["body"]
                if isinstance(body, str):
                    body = json.loads(body)
                
                # Check if body is a dict with opinion field
                if isinstance(body, dict):
                    opinion = body.get("opinion")
                # Or if body contains nested response
                elif isinstance(body, str):
                    # Try to parse as JSON
                    try:
                        body_parsed = json.loads(body)
                        if isinstance(body_parsed, dict):
                            opinion = body_parsed.get("opinion")
                    except:
                        # Try to find JSON in text
                        json_start = body.find("{")
                        json_end = body.rfind("}") + 1
                        if json_start != -1 and json_end > json_start:
                            body_parsed = json.loads(body[json_start:json_end])
                            opinion = body_parsed.get("opinion")
            except (json.JSONDecodeError, TypeError) as e:
                error_message = str(e)
        else:
            # Error response
            error_message = response.get("body", {}).get("error", {}).get("message", f"Status code: {status_code}")
        
        opinions.append({
            "id": custom_id,
            "opinion": opinion,
            "error": error_message if error_message else None,
        })
    
    # Generate summary
    opinion_counts = {
        -2: 0,
        -1: 0,
        0: 0,
        1: 0,
        2: 0,
        "cannot tell": 0,
        None: 0,
    }
    
    for op in opinions:
        opinion = op["opinion"]
        if opinion in opinion_counts:
            opinion_counts[opinion] += 1
        else:
            opinion_counts[None] += 1
    
    total = len(opinions)
    error_count = sum(1 for op in opinions if op.get("error") is not None)
    
    summary = {
        "total_analyzed": total,
        "successful": total - error_count,
        "errors": error_count,
        "opinion_distribution": {
            "-2 (strongly negative)": opinion_counts[-2],
            "-1 (mildly negative)": opinion_counts[-1],
            "0 (neutral)": opinion_counts[0],
            "1 (mildly positive)": opinion_counts[1],
            "2 (strongly positive)": opinion_counts[2],
            "cannot tell": opinion_counts["cannot tell"],
            "invalid": opinion_counts[None],
        },
        "token_usage": all_usage,
    }
    
    # Calculate percentages
    if total > 0:
        summary["opinion_percentages"] = {
            "-2": round(opinion_counts[-2] / total * 100, 2),
            "-1": round(opinion_counts[-1] / total * 100, 2),
            "0": round(opinion_counts[0] / total * 100, 2),
            "1": round(opinion_counts[1] / total * 100, 2),
            "2": round(opinion_counts[2] / total * 100, 2),
            "cannot tell": round(opinion_counts["cannot tell"] / total * 100, 2),
            "invalid": round(opinion_counts[None] / total * 100, 2),
        }
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(SENTIMENT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_df = pd.DataFrame(opinions)
    results_file = output_dir / f"batch_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"Saved detailed results to {results_file}")
    
    # Save summary
    summary_file = output_dir / f"batch_summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_file}")
    
    # Print summary
    print("\n=== Batch Analysis Summary ===")
    print(f"Total tweets analyzed: {summary['total_analyzed']}")
    print(f"Successful: {summary['successful']}")
    print(f"Errors: {summary['errors']}")
    print("\nOpinion Distribution:")
    for key, value in summary["opinion_distribution"].items():
        pct = summary.get("opinion_percentages", {}).get(key.replace(" (strongly negative)", "").replace(" (mildly negative)", "").replace(" (neutral)", "").replace(" (mildly positive)", "").replace(" (strongly positive)", ""), 0)
        print(f"  {key}: {value} ({pct}%)")
    print("\nToken Usage:")
    print(f"  Input tokens: {all_usage['input_tokens']:,}")
    print(f"  Cached tokens: {all_usage['cached_tokens']:,}")
    print(f"  Output tokens: {all_usage['output_tokens']:,}")
    print(f"  Reasoning tokens: {all_usage['reasoning_tokens']:,}")
    print(f"  Total tokens: {all_usage['total_tokens']:,}")
    if all_usage['total_tokens'] > 0:
        print(f"\nToken Efficiency:")
        print(f"  Cache hit rate: {all_usage['cached_tokens'] / all_usage['input_tokens'] * 100:.2f}%" if all_usage['input_tokens'] > 0 else "  Cache hit rate: N/A")
    print(f"\nResults saved to: {output_dir}")
    
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


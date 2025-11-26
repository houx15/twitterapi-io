"""
Sentiment Analysis for Twitter Data using GPT-4.1 API

This script samples tweets from parquet files and analyzes their sentiment
towards AI technology using OpenAI's GPT-4.1 API.

Usage:
    # Analyze with default settings (1000 samples)
    python sentiment_analysis.py analyze

    # Analyze with custom sample size
    python sentiment_analysis.py analyze --sample_size 500

    # Analyze specific parquet file
    python sentiment_analysis.py analyze --input_file parquet_data/tweets_2025-03-20.parquet
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import fire

from config import (
    PARQUET_OUTPUT_DIR,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    SENTIMENT_OUTPUT_DIR,
    SENTIMENT_SAMPLE_SIZE,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Sentiment analyzer for Twitter data using OpenAI API."""

    def __init__(self, api_key: str = OPENAI_API_KEY, model: str = OPENAI_MODEL):
        """
        Initialize the sentiment analyzer.

        Args:
            api_key: OpenAI API key
            model: Model to use for analysis
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"Initialized SentimentAnalyzer with model: {model}")

    def _build_prompt(self, row: pd.Series) -> str:
        """
        Build analysis prompt from tweet data.

        Args:
            row: Tweet data row

        Returns:
            Constructed prompt string
        """
        tweet_text = row.get("text", "") or ""
        author_name = row.get("author.userName", "") or ""
        is_reply = row.get("isReply", False)
        like_count = row.get("likeCount", 0) or 0
        retweet_count = row.get("retweetCount", 0) or 0
        created_at = row.get("createdAt", "") or ""

        # Build content parts
        content_parts = []

        if tweet_text:
            content_parts.append(f"Tweet Text: {tweet_text}")

        if author_name:
            content_parts.append(f"Author: @{author_name}")

        if created_at:
            content_parts.append(f"Posted: {created_at}")

        content_parts.append(f"Is Reply: {is_reply}")
        content_parts.append(f"Likes: {like_count}, Retweets: {retweet_count}")

        content = "\n".join(content_parts)

        prompt = f"""Analyze the following Twitter/X post content and determine its sentiment towards AI technology.

{content}

Based on the content above, classify the sentiment towards AI technology into one of the following categories:
- "positive": The content expresses positive views, enthusiasm, or support for AI technology
- "negative": The content expresses negative views, concerns, or criticism about AI technology
- "neutral": The content mentions AI but does not express a clear positive or negative stance
- "cannot tell": The content does not contain enough information to determine sentiment towards AI technology

Respond with ONLY a JSON object in this exact format:
{{
    "sentiment": "positive" or "negative" or "neutral" or "cannot tell",
    "reason": "Brief explanation of why you classified it this way"
}}"""


english_prompt = """
You will read a piece of text posted by a user on Twitter. Please analyze the text's opinion towards AI technology. Your output must be a JSON object containing only one field, "opinion", whose value must be one of: -2, -1, 0, 1, 2, or "cannot tell".

Label definitions:
- 2 = The text expresses a strongly positive attitude toward AI technology, clearly asserting that the benefits of AI outweigh the harms and that AI brings significant positive impacts to society (e.g., improving convenience, enhancing health and medical services, creating economic opportunities, increasing learning or work efficiency, improving safety, or supporting research and innovation), or expresses reliance on or trust in AI.
- 1 = The text is overall positive toward AI, but the attitude is mild or includes reservations (e.g., expresses support but without strong enthusiasm; believes AI is “generally beneficial” while also mentioning risks or limitations; expresses expectation, interest, or positive impressions, but not strong praise).
- 0 = The text is neutral or has no clear evaluative direction (e.g., mentions both pros and cons but without a clear leaning).
- -1 = The text is overall negative toward AI, but the attitude is mild or includes reservations (e.g., expresses concerns or opposition but does not fully reject AI; believes AI is “risky or harmful” but acknowledges certain benefits; expresses caution, uneasiness, or negative views, but without strong condemnation).
- -2 = The text expresses a strongly negative attitude toward AI technology, clearly asserting that the harms outweigh the benefits and that AI brings significant negative impacts to society (e.g., leading to unemployment, contributing to economic bubbles, increasing privacy risks, reinforcing bias against marginalized groups, generating misinformation or rumors, or posing safety threats), or expresses resistance to or distrust in AI.
- "cannot tell" = The text does not express any attitude toward AI (e.g., content is unrelated to AI or does not reflect a viewpoint).

Please return only a JSON object, for example:
{"opinion": 1}
"""

        return prompt

    def analyze_single(self, row: pd.Series) -> Dict[str, Any]:
        """
        Analyze sentiment for a single tweet.

        Args:
            row: Tweet data row

        Returns:
            Analysis result dictionary
        """
        prompt = self._build_prompt(row)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in sentiment analysis for social media content, particularly regarding technology topics. Provide accurate, concise analysis.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        response_text = response.choices[0].message.content.strip()

        # Try to extract JSON
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                analysis = json.loads(json_text)
                sentiment = analysis.get("sentiment")
                reason = analysis.get("reason", "")
            else:
                sentiment = None
                reason = response_text
        except json.JSONDecodeError:
            sentiment = None
            reason = response_text

        # Normalize sentiment classification
        sentiment = (sentiment or "").lower().strip() or None
        if sentiment and sentiment not in ["positive", "negative", "neutral", "cannot tell"]:
            sentiment = None

        return {
            "sentiment": sentiment,
            "reason": reason,
        }

    def analyze_batch(
        self,
        df: pd.DataFrame,
        delay: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a batch of tweets.

        Args:
            df: DataFrame containing tweets to analyze
            delay: Delay between API calls in seconds

        Returns:
            List of analysis results
        """
        logger.info(f"Starting sentiment analysis for {len(df)} tweets...")

        results = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing tweets"):
            try:
                analysis = self.analyze_single(row)

                result = {
                    "id": row.get("id", ""),
                    "text": row.get("text", ""),
                    "author.userName": row.get("author.userName", ""),
                    "createdAt": row.get("createdAt", ""),
                    "likeCount": row.get("likeCount", 0),
                    "retweetCount": row.get("retweetCount", 0),
                    "sentiment": analysis["sentiment"],
                    "reason": analysis["reason"],
                }
                results.append(result)

                if len(results) % 50 == 0:
                    logger.info(f"Analyzed {len(results)}/{len(df)} tweets")

                # Add delay to avoid API rate limits
                time.sleep(delay)

            except Exception as e:
                logger.error(f"Analysis failed for index {idx}: {e}")
                results.append(
                    {
                        "id": row.get("id", ""),
                        "text": row.get("text", ""),
                        "author.userName": row.get("author.userName", ""),
                        "createdAt": row.get("createdAt", ""),
                        "likeCount": row.get("likeCount", 0),
                        "retweetCount": row.get("retweetCount", 0),
                        "sentiment": None,
                        "reason": "",
                    }
                )

        logger.info(f"Analysis complete: {len(results)} results")
        return results


def load_parquet_files(
    input_dir: str = PARQUET_OUTPUT_DIR,
    input_file: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load tweets from parquet files.

    Args:
        input_dir: Directory containing parquet files
        input_file: Optional specific file to load

    Returns:
        DataFrame containing all tweets
    """
    if input_file:
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        logger.info(f"Loading from {input_file}")
        return pd.read_parquet(input_path, columns=["id", "text"])

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    parquet_files = sorted(input_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    logger.info(f"Found {len(parquet_files)} parquet files")

    dfs = []
    for pf in tqdm(parquet_files, desc="Loading parquet files"):
        dfs.append(pd.read_parquet(pf, columns=["id", "text"]))

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined)} total tweets")

    return combined


def sample_tweets(
    df: pd.DataFrame,
    sample_size: int = SENTIMENT_SAMPLE_SIZE,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Randomly sample tweets from DataFrame.

    Args:
        df: DataFrame containing tweets
        sample_size: Number of tweets to sample
        seed: Random seed for reproducibility

    Returns:
        Sampled DataFrame
    """
    if len(df) <= sample_size:
        logger.warning(
            f"DataFrame has {len(df)} rows, less than sample size {sample_size}"
        )
        return df

    sampled = df.sample(n=sample_size, random_state=seed)
    logger.info(f"Sampled {len(sampled)} tweets from {len(df)} total")

    return sampled


def generate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary statistics from analysis results.

    Args:
        results: List of analysis results

    Returns:
        Summary dictionary
    """
    sentiments = [r["sentiment"] for r in results if r.get("sentiment")]

    summary = {
        "total_analyzed": len(results),
        "positive": sentiments.count("positive"),
        "negative": sentiments.count("negative"),
        "neutral": sentiments.count("neutral"),
        "cannot_tell": sentiments.count("cannot tell"),
    }

    # Calculate percentages
    total = summary["total_analyzed"]
    if total > 0:
        summary["positive_pct"] = round(summary["positive"] / total * 100, 2)
        summary["negative_pct"] = round(summary["negative"] / total * 100, 2)
        summary["neutral_pct"] = round(summary["neutral"] / total * 100, 2)
        summary["cannot_tell_pct"] = round(summary["cannot_tell"] / total * 100, 2)

    return summary


def analyze(
    input_file: Optional[str] = None,
    input_dir: str = PARQUET_OUTPUT_DIR,
    output_dir: str = SENTIMENT_OUTPUT_DIR,
    sample_size: int = SENTIMENT_SAMPLE_SIZE,
    seed: int = 42,
    delay: float = 0.5,
):
    """
    Run sentiment analysis on sampled tweets.

    Args:
        input_file: Optional specific parquet file to analyze
        input_dir: Directory containing parquet files
        output_dir: Directory for output results
        sample_size: Number of tweets to sample
        seed: Random seed for reproducibility
        delay: Delay between API calls in seconds
    """
    logger.info("=== Starting Sentiment Analysis ===")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_parquet_files(input_dir=input_dir, input_file=input_file)

    # Sample tweets
    sampled_df = sample_tweets(df, sample_size=sample_size, seed=seed)

    # Initialize analyzer
    analyzer = SentimentAnalyzer()

    # Run analysis
    results = analyzer.analyze_batch(sampled_df, delay=delay)

    # Generate summary
    summary = generate_summary(results)

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Save detailed results as CSV
    results_df = pd.DataFrame(results)
    results_file = output_path / f"sentiment_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"Saved detailed results to {results_file}")

    # Save summary as JSON
    summary_file = output_path / f"sentiment_summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_file}")

    # Print summary
    print("\n=== Sentiment Analysis Summary ===")
    print(f"Total tweets analyzed: {summary['total_analyzed']}")
    print(f"Positive: {summary['positive']} ({summary.get('positive_pct', 0)}%)")
    print(f"Negative: {summary['negative']} ({summary.get('negative_pct', 0)}%)")
    print(f"Neutral: {summary['neutral']} ({summary.get('neutral_pct', 0)}%)")
    print(
        f"Cannot Tell: {summary['cannot_tell']} ({summary.get('cannot_tell_pct', 0)}%)"
    )
    print(f"\nResults saved to: {output_path}")

    logger.info("=== Sentiment Analysis Complete ===")


def main():
    """Main entry point using fire package."""
    fire.Fire(
        {
            "analyze": analyze,
        }
    )


if __name__ == "__main__":
    main()

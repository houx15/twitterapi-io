"""
脚本用于更新旧的 state 文件，添加 max_id 字段并重置 finished 状态。

这个脚本会：
1. 读取旧的 state 文件
2. 将所有 finished 改为 false
3. 从对应的 jsonl 文件中读取最后一个 tweet 的 id，更新 max_id 字段
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from config import STATE_FILE, DATA_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def get_last_tweet_id_from_jsonl(jsonl_file: Path) -> Optional[str]:
    """
    从 JSONL 文件中读取最后一个 tweet 的 id。

    Args:
        jsonl_file: JSONL 文件路径

    Returns:
        最后一个 tweet 的 id，如果文件不存在或为空则返回 None
    """
    if not jsonl_file.exists():
        logger.warning(f"JSONL file not found: {jsonl_file}")
        return None

    try:
        last_tweet_id = None
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    tweet = json.loads(line)
                    tweet_id = tweet.get("id")
                    if tweet_id:
                        last_tweet_id = tweet_id
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line in {jsonl_file}: {e}")
                    continue

        return last_tweet_id
    except Exception as e:
        logger.error(f"Error reading JSONL file {jsonl_file}: {e}")
        return None


def update_state_file(state_file: str = STATE_FILE, data_dir: str = DATA_DIR) -> None:
    """
    更新 state 文件，添加 max_id 字段并重置 finished 状态。

    Args:
        state_file: State 文件路径
        data_dir: 数据目录路径，包含 JSONL 文件
    """
    state_path = Path(state_file)
    data_path = Path(data_dir)

    if not state_path.exists():
        logger.warning(f"State file not found: {state_path}")
        return

    # 读取旧的 state
    logger.info(f"Reading state file: {state_path}")
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    logger.info(f"Found {len(state)} dates in state file")

    # 更新每个日期的状态
    updated_count = 0
    for date, date_state in state.items():
        logger.info(f"Processing date: {date}")

        # 重置 finished 状态
        date_state["finished"] = False

        # 从 JSONL 文件中获取最后一个 tweet 的 id
        jsonl_file = data_path / f"tweets_{date}.jsonl"
        last_tweet_id = get_last_tweet_id_from_jsonl(jsonl_file)

        if last_tweet_id:
            date_state["max_id"] = last_tweet_id
            logger.info(f"  Updated max_id: {last_tweet_id}")
        else:
            # 如果没有找到 JSONL 文件或文件为空，清除 max_id
            if "max_id" in date_state:
                del date_state["max_id"]
            logger.info(f"  No max_id found (JSONL file may be empty or missing)")

        updated_count += 1

    # 保存更新后的 state
    backup_path = state_path.with_suffix(".json.backup")
    logger.info(f"Creating backup: {backup_path}")
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    logger.info(f"Saving updated state to: {state_path}")
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    logger.info(f"Successfully updated {updated_count} dates in state file")
    logger.info(f"Backup saved to: {backup_path}")


def get_min_tweet_id_by_date_from_jsonl(jsonl_file: Path, target_date: str) -> Optional[str]:
    """
    从 JSONL 文件中读取所有 tweets，找到 created_at 日期匹配目标日期的 tweets 中的最小 id。

    Args:
        jsonl_file: JSONL 文件路径
        target_date: 目标日期，格式为 "YYYY-MM-DD"

    Returns:
        匹配日期的 tweets 中的最小 id，如果没有找到则返回 None
    """
    if not jsonl_file.exists():
        logger.warning(f"JSONL file not found: {jsonl_file}")
        return None, None

    try:
        target_date_obj = datetime.strptime(target_date, "%Y-%m-%d").date()
        min_id = None
        min_created_at = None

        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    tweet = json.loads(line)
                    tweet_id = tweet.get("id")
                    created_at = tweet.get("createdAt")

                    if not tweet_id or not created_at:
                        continue

                    # 解析 created_at 日期
                    try:
                        created_at_date = datetime.strptime(
                            created_at, "%a %b %d %H:%M:%S +0000 %Y"
                        )
                        if created_at_date.date() == target_date_obj:
                            if (min_id is None) or (int(tweet_id) < int(min_id)):
                                min_id = tweet_id
                                min_created_at = created_at
                    except ValueError as e:
                        logger.warning(
                            f"Failed to parse created_at '{created_at}' in {jsonl_file}: {e}"
                        )
                        continue

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line in {jsonl_file}: {e}")
                    continue

        return min_id, min_created_at

    except Exception as e:
        logger.error(f"Error reading JSONL file {jsonl_file}: {e}")
        return None, None


def update_max_id_with_date_filter(
    state_file: str = STATE_FILE, data_dir: str = DATA_DIR
) -> None:
    """
    更新 state 文件，针对每一天，将 max_id 更新为当天所有数据中，
    确实为当天日期（created_at 日期对得上）中的最小一个 id。
    同时清空所有 current_cursor。

    Args:
        state_file: State 文件路径
        data_dir: 数据目录路径，包含 JSONL 文件
    """
    state_path = Path(state_file)
    data_path = Path(data_dir)

    if not state_path.exists():
        logger.warning(f"State file not found: {state_path}")
        return

    # 读取旧的 state
    logger.info(f"Reading state file: {state_path}")
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    logger.info(f"Found {len(state)} dates in state file")

    # 更新每个日期的状态
    updated_count = 0
    for date, date_state in state.items():
        logger.info(f"Processing date: {date}")

        # 从 JSONL 文件中获取匹配日期的 tweets 中的最小 id
        jsonl_file = data_path / f"tweets_{date}.jsonl"
        min_tweet_id, min_created_at = get_min_tweet_id_by_date_from_jsonl(jsonl_file, date)

        if min_tweet_id:
            date_state["max_id"] = min_tweet_id
            date_state["until_time_str"] = min_created_at
            logger.info(f"  Updated max_id to minimum matching id: {min_tweet_id}")
        else:
            # 如果没有找到匹配的 tweet，清除 max_id
            if "max_id" in date_state:
                del date_state["max_id"]
            logger.info(f"  No matching tweets found for date {date}")

        # 清空 current_cursor
        # if "current_cursor" in date_state:
        #     del date_state["current_cursor"]
        #     logger.info(f"  Cleared current_cursor for date {date}")

        updated_count += 1

    # 保存更新后的 state
    backup_path = state_path.with_suffix(".json.backup")
    logger.info(f"Creating backup: {backup_path}")
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    logger.info(f"Saving updated state to: {state_path}")
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    logger.info(f"Successfully updated {updated_count} dates in state file")
    logger.info(f"Backup saved to: {backup_path}")


def test():
    jsonl_file = Path(DATA_DIR, "tweets_2025-03-01.jsonl")
    min_tweet_id = get_min_tweet_id_by_date_from_jsonl(jsonl_file, "2025-03-01")
    print(min_tweet_id)

    jsonl_file = Path(DATA_DIR, "tweets_2025-02-20.jsonl")
    min_tweet_id = get_min_tweet_id_by_date_from_jsonl(jsonl_file, "2025-02-20")
    print(min_tweet_id)

if __name__ == "__main__":
    import fire

    fire.Fire({
        "update1": update_state_file,
        "update2": update_max_id_with_date_filter,
        "test": test,
    })

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


if __name__ == "__main__":
    import fire

    fire.Fire(update_state_file)

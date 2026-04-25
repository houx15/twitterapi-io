"""
将 user_tweet_crawler.py 生成的 JSONL 文件转换为 Parquet 格式。

此脚本读取 user_tweet_crawler.py 生成的 tweets_XXXXXX.jsonl 文件，
根据 tweet ID 全局去重，并转换为 Parquet 格式。

Usage:
    # 转换指定 topic 的 JSONL 文件
    python convert_user_tweets_to_parquet.py convert --topic abortion

    # 使用默认 topic (abortion)
    python convert_user_tweets_to_parquet.py convert
"""

import json
import logging
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm
import fire
import re

logger = logging.getLogger(__name__)


def load_config(config_name: str = "config"):
    """
    动态加载配置模块。
    
    Args:
        config_name: 配置模块名称或配置文件路径 (default: "config")
                    可以是:
                    - 模块名: "config", "config_custom", 等
                    - 文件路径: "config.xxx.py", "configs/xxx.py", "./configs/config.xxx.py", 等
                    - 绝对路径: "/path/to/config.xxx.py"
    
    Returns:
        配置模块对象
    """
    # 检查是否是文件路径（包含 .py 或 /）
    if config_name.endswith('.py') or '/' in config_name or '\\' in config_name:
        # 从文件路径加载
        config_path = Path(config_name)
        if not config_path.is_absolute():
            # 如果是相对路径，尝试相对于脚本目录查找
            script_dir = Path(__file__).parent
            config_path = script_dir / config_name
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # 基于文件路径生成唯一的模块名
        module_name = f"config_{config_path.stem}_{hash(str(config_path))}"
        
        # 从文件加载模块
        spec = importlib.util.spec_from_file_location(module_name, config_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create spec for config file: {config_path}")
        
        config_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = config_module
        spec.loader.exec_module(config_module)
        
        logger.info(f"Loaded configuration from file: {config_path}")
        return config_module
    else:
        # 作为模块名加载
        try:
            config_module = importlib.import_module(config_name)
            logger.info(f"Loaded configuration from module: {config_name}")
            return config_module
        except ImportError as e:
            logger.error(f"Failed to import config module '{config_name}': {e}")
            raise


def setup_logging(config):
    """使用配置模块中的配置设置日志。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_FILE),
            logging.StreamHandler()
        ],
    )


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
    从 tweet 对象中提取所需列。

    Args:
        tweet: 来自 JSONL 的原始 tweet 字典

    Returns:
        仅包含所需列的字典
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
    从 JSONL 文件读取 tweets。

    Args:
        file_path: JSONL 文件路径

    Returns:
        tweet 字典列表
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


def is_user_tweet_file(filename: str) -> bool:
    """
    判断文件是否是 user_tweet_crawler.py 生成的文件格式。
    格式: tweets_XXXXXX.jsonl (6位数字)

    Args:
        filename: 文件名

    Returns:
        如果是 user tweet 文件格式返回 True
    """
    # 匹配 tweets_XXXXXX.jsonl 格式（6位数字）
    pattern = r'^tweets_\d{6}\.jsonl$'
    return bool(re.match(pattern, filename))


def convert_file_to_dataframe(input_file: Path) -> pd.DataFrame:
    """
    将单个 JSONL 文件转换为 DataFrame。

    Args:
        input_file: 输入 JSONL 文件路径

    Returns:
        包含提取数据的 DataFrame
    """
    tweets = read_jsonl_file(input_file)
    
    if not tweets:
        return pd.DataFrame()
    
    # 提取所需列
    extracted = [extract_tweet_data(tweet) for tweet in tweets]
    
    # 创建 DataFrame
    df = pd.DataFrame(extracted)
    
    # 移除 id 为 None 的行
    df = df[df["id"].notna()].copy()
    
    return df


def convert(
    topic: str = "abortion",
):
    """
    将 user_tweet_crawler.py 生成的 JSONL 文件转换为 Parquet 格式，并全局去重。

    每个文件处理完后立即保存，使用 set 跟踪已见过的 tweet id 进行全局去重。

    Args:
        topic: 要转换的 topic（默认: "abortion"），用于加载对应的配置文件
    """
    # 加载配置文件
    config_name = f"configs/{topic}.py"
    config = load_config(config_name)
    setup_logging(config)
    logger.info(f"=== Starting conversion for {topic} ===")

    # 从 config 读取目录
    input_dir = config.DATA_DIR
    output_dir = config.PARQUET_OUTPUT_DIR

    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 查找所有 user tweet 文件（tweets_XXXXXX.jsonl 格式）
    all_jsonl_files = sorted(input_path.glob("tweets_*.jsonl"))
    jsonl_files = [f for f in all_jsonl_files if is_user_tweet_file(f.name)]

    if not jsonl_files:
        logger.warning(f"在 {input_dir} 中未找到 user tweet 文件（tweets_XXXXXX.jsonl 格式）")
        return

    logger.info(f"找到 {len(jsonl_files)} 个 user tweet 文件需要转换")

    # 使用 set 跟踪已见过的 tweet id
    seen_ids = set()
    total_lines = 0
    total_unique_tweets = 0
    total_duplicates_removed = 0
    files_saved = 0

    # 逐个文件处理
    for jsonl_file in tqdm(jsonl_files, desc="处理文件"):
        df = convert_file_to_dataframe(jsonl_file)
        
        if df.empty:
            continue
        
        initial_count = len(df)
        total_lines += initial_count
        
        # 过滤掉已见过的 tweet id
        df_new = df[~df["id"].isin(seen_ids)].copy()
        duplicates_in_file = initial_count - len(df_new)
        total_duplicates_removed += duplicates_in_file
        
        # 将新的 tweet id 添加到 set 中
        new_ids = set(df_new["id"].dropna().unique())
        seen_ids.update(new_ids)
        total_unique_tweets += len(new_ids)
        
        # 如果有新的 tweets，保存为 Parquet
        if not df_new.empty:
            output_file = output_path / jsonl_file.name.replace(".jsonl", ".parquet")
            df_new.to_parquet(
                output_file,
                engine="fastparquet",
                compression="snappy",
                index=False
            )
            files_saved += 1
            logger.info(
                f"{jsonl_file.name}: {initial_count} 条 -> {len(df_new)} 条 unique "
                f"(移除 {duplicates_in_file} 条重复)"
            )
        else:
            logger.info(f"{jsonl_file.name}: 所有 {initial_count} 条 tweets 都是重复的，跳过保存")

    # 打印统计信息
    print("\n" + "=" * 50)
    print("转换摘要")
    print("=" * 50)
    print(f"Topic: {topic}")
    print(f"处理的文件数: {len(jsonl_files)}")
    print(f"保存的文件数: {files_saved}")
    print(f"读取的总行数: {total_lines}")
    print(f"全局去重后的 unique tweets 数: {total_unique_tweets}")
    print(f"移除的重复 tweets 数: {total_duplicates_removed}")
    print(f"输出目录: {output_path}")
    print("=" * 50)
    
    logger.info(f"=== Conversion for {topic} completed ===")


def main():
    """主入口点，使用 fire 包。"""
    fire.Fire({
        "convert": convert,
    })


if __name__ == "__main__":
    main()


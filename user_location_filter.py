"""
for us location filter
go to adroit bot-detection folder
"""

import json
import logging
from pathlib import Path
from typing import Set
import pandas as pd
from tqdm import tqdm
import fire

from config import PARQUET_OUTPUT_DIR, USER_LOCATION_FILTER_OUTPUT_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Use USER_LOCATION_FILTER_OUTPUT_DIR as base_dir for output files
# Use PARQUET_OUTPUT_DIR for reading parquet files
BASE_DIR = Path(USER_LOCATION_FILTER_OUTPUT_DIR)
BASE_DIR.mkdir(parents=True, exist_ok=True)


def export_user_location_json():
    """
    based on the parquet from convert_to_parquet.py, export the user locations json to a list, use set to dedup
    save to base_dir found in config.py
    """
    logger.info("开始导出用户位置信息...")
    
    # 从 PARQUET_OUTPUT_DIR 读取 parquet 文件
    parquet_dir = Path(PARQUET_OUTPUT_DIR)
    parquet_files = list(parquet_dir.glob("*.parquet"))
    
    if not parquet_files:
        logger.error(f"在 {parquet_dir} 中未找到 parquet 文件")
        return
    
    logger.info(f"找到 {len(parquet_files)} 个 parquet 文件")
    
    # 使用 set 去重存储所有位置
    locations_set: Set[str] = set()
    
    # 遍历所有 parquet 文件
    for parquet_file in tqdm(parquet_files, desc="处理 parquet 文件"):
        try:
            df = pd.read_parquet(parquet_file, engine="fastparquet")
            
            # 提取 author.location 列，去除空值
            if "author.location" in df.columns:
                locations = df["author.location"].dropna().unique()
                # 转换为字符串并添加到集合中
                for loc in locations:
                    if loc and str(loc).strip():
                        locations_set.add(str(loc).strip())
        except Exception as e:
            logger.warning(f"处理文件 {parquet_file} 时出错: {e}")
            continue
    
    # 转换为列表并排序
    locations_list = sorted(list(locations_set))
    
    # 保存到 JSON 文件
    output_file = BASE_DIR / "user_locations.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(locations_list, f, ensure_ascii=False, indent=2)
    
    logger.info(f"成功导出 {len(locations_list)} 个唯一位置到 {output_file}")


def generate_us_userids():
    """
    try to load from base_dir/us_user_analysis.json, if not found, raise error
    {
                "non_us": list(non_us),
                "us": list(us),
                "not_sure": list(not_sure),
                "undecided": list(undecided),
            }
    based on the parquet from convert_to_parquet.py, generate a user id set, whose location is in the us list, save to base_dir/us_userids.json
    """
    logger.info("开始生成美国用户 ID...")
    
    # 加载用户位置分析结果
    analysis_file = BASE_DIR / "us_user_analysis.json"
    if not analysis_file.exists():
        raise FileNotFoundError(f"未找到分析文件: {analysis_file}")
    
    with open(analysis_file, "r", encoding="utf-8") as f:
        analysis_data = json.load(f)
    
    # 获取美国位置列表
    us_locations = set(analysis_data.get("us", []))
    
    if not us_locations:
        logger.warning("us 列表为空，将不会匹配任何用户")
    
    logger.info(f"加载了 {len(us_locations)} 个美国位置")
    
    # 从 PARQUET_OUTPUT_DIR 读取 parquet 文件
    parquet_dir = Path(PARQUET_OUTPUT_DIR)
    parquet_files = list(parquet_dir.glob("*.parquet"))
    
    if not parquet_files:
        logger.error(f"在 {parquet_dir} 中未找到 parquet 文件")
        return
    
    logger.info(f"找到 {len(parquet_files)} 个 parquet 文件")
    
    # 使用 set 存储美国用户 ID
    us_userids_set: Set[str] = set()
    
    # 遍历所有 parquet 文件
    for parquet_file in tqdm(parquet_files, desc="处理 parquet 文件"):
        try:
            df = pd.read_parquet(parquet_file, engine="fastparquet")
            
            # 检查必要的列是否存在
            if "author.location" not in df.columns or "author.id" not in df.columns:
                logger.warning(f"文件 {parquet_file} 缺少必要的列")
                continue
            
            # 筛选位置在美国的用户
            us_mask = df["author.location"].notna() & df["author.location"].isin(us_locations)
            us_users = df[us_mask]
            
            # 提取用户 ID，去除空值
            user_ids = us_users["author.id"].dropna().unique()
            for uid in user_ids:
                if uid:
                    us_userids_set.add(str(uid))
                    
        except Exception as e:
            logger.warning(f"处理文件 {parquet_file} 时出错: {e}")
            continue
    
    # 转换为列表并排序
    us_userids_list = sorted(list(us_userids_set))
    
    # 保存到 JSON 文件
    output_file = BASE_DIR / "us_userids.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(us_userids_list, f, ensure_ascii=False, indent=2)
    
    logger.info(f"成功生成 {len(us_userids_list)} 个美国用户 ID 到 {output_file}")

if __name__ == "__main__":
    fire.Fire({
        "export": export_user_location_json,
        "generate": generate_us_userids,
    })
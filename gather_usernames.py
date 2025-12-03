import json
import glob
import os

root_dir = "/mnt/disk-ccc-covid3-llm/tweet_crawler/clients"
files = glob.glob(os.path.join(root_dir, "*/user_screenname_map.json"))

target_file = "/mnt/disk-ccc-covid3-llm/covid3/usernames.txt"

all_usernames = set()
for file in files:
    with open(file, "r") as f:
        data = json.load(f)
        all_usernames.update(data.values())

with open(target_file, "w", encoding="utf-8") as f:
    f.write("\n".join(all_usernames))
import subprocess
import os
from pathlib import Path
import json


def sync_r2_bucket(
        cf_account_id: str,
        r2_access_key: str,
        r2_secret_access_key: str,
        path_to_dir: Path, 
        # path_rclone_config: str, 
        bucket_name: str,  path_in_bucket: str="", timeout: int=300,
        exclude_patterns: list[str]=[],
        ) -> None:
        
    # TODO: add timeout
    base_env = os.environ.copy()
    path_to_dir = Path(path_to_dir)
    # path_rclone_config = Path(path_rclone_config)

    if not path_in_bucket:
        path_in_bucket = path_to_dir.name
    else:
        path_in_bucket = str(path_in_bucket)

    # subprocess.run(["rclone", "--config", path_rclone_config, "tree", f"r2_cayleypy:{bucket_name}"], env=base_env)

    command = [
        "rclone",
        "--s3-provider", "Cloudflare",
        "--s3-endpoint", f"https://{cf_account_id}.r2.cloudflarestorage.com",
        "--s3-access-key-id", r2_access_key,
        "--s3-secret-access-key", r2_secret_access_key,
        "sync", str(path_to_dir), f":s3:{bucket_name}/{path_in_bucket}",
    ]

    for pattern in exclude_patterns:
        command = command + ["--exclude", pattern]

    print("Running command:", " ".join(command))
    subprocess.run(command, env=base_env, timeout=timeout)
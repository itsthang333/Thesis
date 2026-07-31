from __future__ import annotations

"""Resume an official Kaggle kernel-output inventory with atomic file writes."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
from typing import Optional, Pattern

import requests


@dataclass(frozen=True)
class OutputItem:
    relative: str
    url: str


def safe_target(destination: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"Absolute Kaggle output path: {relative}")
    root = destination.resolve()
    target = (destination / candidate).resolve()
    if target == root or root not in target.parents:
        raise ValueError(f"Kaggle output path escapes destination: {relative}")
    return target


def list_official_output(
    kernel: str,
    *,
    page_size: int = 100,
) -> tuple[list[OutputItem], str, str]:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    request_type = api.kernels_output.__globals__["ApiListKernelSessionOutputRequest"]
    owner, slug, _version = api.parse_kernel_string(kernel)
    token = None
    items: list[OutputItem] = []
    log = ""
    with api.build_kaggle_client() as client:
        while True:
            request = request_type()
            request.user_name = owner
            request.kernel_slug = slug
            request.page_size = page_size
            if token:
                request.page_token = token
            response = client.kernels.kernels_api_client.list_kernel_session_output(
                request
            )
            if not log and response.log:
                log = response.log
            items.extend(
                OutputItem(relative=item.file_name, url=item.url)
                for item in response.files or []
            )
            token = response.next_page_token
            if not token:
                break
    if len(items) != len({item.relative for item in items}):
        raise RuntimeError("Official Kaggle output inventory contains duplicate paths")
    return items, log, slug


def select_items(
    items: list[OutputItem],
    include: Optional[Pattern[str]],
) -> list[OutputItem]:
    if include is None:
        return items
    return [item for item in items if include.search(item.relative)]


def download_item(
    item: OutputItem,
    destination: Path,
    *,
    max_attempts: int,
    connect_timeout: float = 30.0,
    read_timeout: float = 120.0,
) -> str:
    target = safe_target(destination, item.relative)
    if target.is_file():
        return "existing"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    for attempt in range(1, max_attempts + 1):
        try:
            with requests.get(
                item.url,
                stream=True,
                timeout=(connect_timeout, read_timeout),
            ) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            handle.write(block)
            os.replace(partial, target)
            return "downloaded"
        except (OSError, requests.RequestException):
            if partial.exists():
                partial.unlink()
            if attempt == max_attempts:
                raise
            time.sleep(float(attempt))
    raise AssertionError("unreachable")


def write_log_atomic(destination: Path, slug: str, log: str) -> Path:
    path = destination / f"{slug}.log"
    partial = path.with_name(path.name + ".part")
    partial.write_text(log, encoding="utf-8")
    os.replace(partial, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--include-regex")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be between 1 and 16")
    if not 1 <= args.max_attempts <= 10:
        raise ValueError("max-attempts must be between 1 and 10")
    include = re.compile(args.include_regex) if args.include_regex else None
    args.destination.mkdir(parents=True, exist_ok=True)
    items, log, slug = list_official_output(args.kernel)
    selected = select_items(items, include)
    outcomes = {"existing": 0, "downloaded": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                download_item,
                item,
                args.destination,
                max_attempts=args.max_attempts,
            )
            for item in selected
        ]
        for future in as_completed(futures):
            outcomes[future.result()] += 1
    log_path = write_log_atomic(args.destination, slug, log)
    remaining_parts = list(args.destination.rglob("*.part"))
    if remaining_parts:
        raise RuntimeError(f"Partial output files remain: {remaining_parts[:5]}")
    print(
        json.dumps(
            {
                "kernel": args.kernel,
                "official_inventory": len(items),
                "selected_inventory": len(selected),
                "existing": outcomes["existing"],
                "downloaded": outcomes["downloaded"],
                "log_path": str(log_path),
                "log_bytes": log_path.stat().st_size,
                "status_queried": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

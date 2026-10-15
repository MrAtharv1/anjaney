"""Streams ONLY the 4 split CSVs and the 11 KB <id>_TRANSCRIPT.csv from each of the 189 zip files on dcapswoz.ict.usc.edu using HTTP Range requests."""

import os
import shutil
import zipfile
from pathlib import Path

import requests
from remotezip import RemoteZip

BASE_URL = "https://dcapswoz.ict.usc.edu/wwwdaicwoz/"
SPLIT_FILES = [
    "train_split_Depression_AVEC2017.csv",
    "dev_split_Depression_AVEC2017.csv",
    "test_split_Depression_AVEC2017.csv",
    "full_test_split.csv",
]
EXCLUDED_IDS = {342, 394, 398, 460}

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Referer": "https://dcapswoz.ict.usc.edu/wwwdaicwoz/",
    "Accept": "*/*",
}


def harvest_local_downloads(out_root: Path) -> None:
    """Copy any split CSVs or transcripts already downloaded in C:\\Users\\intel\\Downloads."""
    downloads_dir = Path.home() / "Downloads"
    if not downloads_dir.exists():
        return

    for csv_name in SPLIT_FILES:
        src = downloads_dir / csv_name
        dest = out_root / csv_name
        if src.exists() and not dest.exists():
            shutil.copy2(src, dest)
            print(f"[LOCAL] Copied {csv_name} from Downloads")

    for pid in range(300, 493):
        if pid in EXCLUDED_IDS:
            continue
        dest = out_root / f"{pid}_P" / f"{pid}_TRANSCRIPT.csv"
        if dest.exists():
            continue

        # Check extracted folder in Downloads (e.g. Downloads/393_P/393_TRANSCRIPT.csv)
        extracted_candidates = [
            downloads_dir / f"{pid}_P" / f"{pid}_TRANSCRIPT.csv",
            downloads_dir / f"{pid}_P" / f"{pid}_P" / f"{pid}_TRANSCRIPT.csv",
        ]
        for cand in extracted_candidates:
            if cand.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cand, dest)
                print(f"[LOCAL] Copied {pid}_TRANSCRIPT.csv from Downloads/{pid}_P")
                break

        # Check zip archive in Downloads (e.g. Downloads/393_P.zip)
        zip_cand = downloads_dir / f"{pid}_P.zip"
        if not dest.exists() and zip_cand.exists():
            try:
                with zipfile.ZipFile(zip_cand, "r") as zf:
                    matches = [
                        m for m in zf.namelist() if m.upper().endswith(f"{pid}_TRANSCRIPT.CSV")
                    ]
                    if matches:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(zf.read(matches[0]))
                        print(
                            f"[LOCAL ZIP] Extracted {pid}_TRANSCRIPT.csv from Downloads/{pid}_P.zip"
                        )
            except Exception:
                pass


def download_transcripts_only(target_dir: Path | str = "data/daic_woz") -> None:
    out_root = Path(target_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # First grab anything already in C:\Users\intel\Downloads
    harvest_local_downloads(out_root)

    user = os.environ.get("DAIC_USER", "").strip()
    password = os.environ.get("DAIC_PASS", "").strip()
    cookie = os.environ.get("DAIC_COOKIE", "").strip()
    auth = (user, password) if user and password else None

    headers = dict(BROWSER_HEADERS)
    if cookie:
        headers["Cookie"] = cookie

    session = requests.Session()
    session.headers.update(headers)
    if auth:
        session.auth = auth

    # 1. Download the 4 small CSV split files
    for csv_name in SPLIT_FILES:
        dest = out_root / csv_name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[OK] {csv_name} ready ({dest.stat().st_size} bytes)")
            continue
        print(f"Downloading {csv_name}...")
        resp = session.get(f"{BASE_URL}{csv_name}", timeout=30)
        if resp.status_code in (401, 403):
            print(
                f"\n[AUTH REQUIRED] Server returned HTTP {resp.status_code}.\n"
                "Please either:\n"
                "  1) Download the 4 small .csv files at the bottom of your Brave tab into C:\\Users\\intel\\Downloads, OR\n"
                "  2) Set your USC credentials in PowerShell before running:\n"
                '     $env:DAIC_USER="your_username"; $env:DAIC_PASS="your_password"\n'
            )
            return
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        print(f" -> Saved {csv_name} ({len(resp.content)} bytes)")

    # 2. Stream ONLY <id>_TRANSCRIPT.csv (~11 KB) from each <id>_P.zip
    extracted = 0
    for pid in range(300, 493):
        if pid in EXCLUDED_IDS:
            continue
        p_dir = out_root / f"{pid}_P"
        transcript_dest = p_dir / f"{pid}_TRANSCRIPT.csv"
        if transcript_dest.exists() and transcript_dest.stat().st_size > 0:
            extracted += 1
            continue

        zip_url = f"{BASE_URL}{pid}_P.zip"
        try:
            with RemoteZip(zip_url, auth=auth, headers=headers) as rz:
                matches = [
                    m
                    for m in rz.namelist()
                    if m.upper().endswith(f"{pid}_TRANSCRIPT.CSV")
                    and not Path(m).name.startswith("._")
                    and not m.startswith("__MACOSX")
                ]
                if not matches:
                    print(f"[WARN] No {pid}_TRANSCRIPT.csv found inside {pid}_P.zip")
                    continue
                p_dir.mkdir(parents=True, exist_ok=True)
                data = rz.read(matches[0])
                transcript_dest.write_bytes(data)
                extracted += 1
                print(
                    f"[{extracted}/189] Extracted {pid}_P/{pid}_TRANSCRIPT.csv ({len(data) / 1024:.1f} KB)"
                )
        except Exception as exc:
            print(f"[SKIP] {pid}_P.zip: {exc}")

    print(f"\nCompleted! {extracted}/189 session transcripts ready in '{out_root}'.")


if __name__ == "__main__":
    download_transcripts_only()

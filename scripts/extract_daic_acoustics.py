"""Stream and extract window-level acoustic biomarkers from DAIC-WOZ COVAREP and FORMANT files.

Compliance:
- AVEC2017 specifications: 100 Hz sampling (0.01s per frame)
- Slices strictly on SpeakerRole.PARTICIPANT turns (never interviewer turns)
- Interruption intervals (Sessions 373 & 444) and scrubbed frames masked
- Voiced filtering (VUV == 1 & F0 > 0) applied to glottal/prosodic and formant features
- Zero raw audio/video download; temporary frame arrays discarded from memory
- Saves only compact per-participant summary parquet <pid>_ACOUSTIC_WINDOWS.parquet (~4 KB)
- AGENTS.md Rule 1: Zero free-text logging (aggregate counts and IDs only)
"""

import concurrent.futures
import io
import os
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from remotezip import RemoteZip

from anjaney.data.daic_loader import DAICLoader
from anjaney.data.schemas import Session, SpeakerRole
from anjaney.features.acoustic import compute_window_acoustics
from anjaney.preprocess.cleaner import TAG_PATTERN

BASE_URL = "https://dcapswoz.ict.usc.edu/wwwdaicwoz/"
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    ),
    "Referer": "https://dcapswoz.ict.usc.edu/wwwdaicwoz/",
    "Accept": "*/*",
}


def get_participant_turns_from_session(session: Session) -> list[Any]:
    """Retrieve raw unshifted participant turns for a session matching WindowBuilder chunks."""
    raw_part = []
    for t in session.turns:
        clean_txt = TAG_PATTERN.sub("", t.text).strip()
        if clean_txt and t.speaker == SpeakerRole.PARTICIPANT:
            raw_part.append(t)
    return raw_part


def get_session_acoustic_arrays(pid: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Obtain COVAREP and FORMANT frame matrices from local Downloads or RemoteZip."""
    downloads_dir = Path.home() / "Downloads"

    # 1. Check local directory candidates
    local_dir_candidates = [
        downloads_dir / f"{pid}_P",
        downloads_dir / f"{pid}_P" / f"{pid}_P",
    ]
    for d in local_dir_candidates:
        cov_f = d / f"{pid}_COVAREP.csv"
        form_f = d / f"{pid}_FORMANT.csv"
        if cov_f.exists() and form_f.exists():
            try:
                cov = pd.read_csv(cov_f, header=None).values
                form = pd.read_csv(form_f, header=None).values
                return cov, form
            except Exception:
                pass

    # 2. Check local zip candidates
    local_zip = downloads_dir / f"{pid}_P.zip"
    if local_zip.exists():
        try:
            with zipfile.ZipFile(local_zip, "r") as zf:
                cov_entry = next(
                    (m for m in zf.namelist() if m.upper().endswith(f"{pid}_COVAREP.CSV")), None
                )
                form_entry = next(
                    (m for m in zf.namelist() if m.upper().endswith(f"{pid}_FORMANT.CSV")), None
                )
                if cov_entry and form_entry:
                    cov_bytes = zf.read(cov_entry)
                    form_bytes = zf.read(form_entry)
                    cov = pd.read_csv(io.BytesIO(cov_bytes), header=None).values
                    form = pd.read_csv(io.BytesIO(form_bytes), header=None).values
                    return cov, form
        except Exception:
            pass

    # 3. Stream via RemoteZip with retry
    zip_url = f"{BASE_URL}{pid}_P.zip"
    user = os.environ.get("DAIC_USER", "").strip()
    password = os.environ.get("DAIC_PASS", "").strip()
    auth = (user, password) if user and password else None

    for attempt in range(3):
        try:
            with RemoteZip(zip_url, auth=auth, headers=BROWSER_HEADERS, timeout=30) as rz:
                cov_entry = next(
                    (m for m in rz.namelist() if m.upper().endswith(f"{pid}_COVAREP.CSV")), None
                )
                form_entry = next(
                    (m for m in rz.namelist() if m.upper().endswith(f"{pid}_FORMANT.CSV")), None
                )
                if not cov_entry or not form_entry:
                    return None, None

                cov_bytes = rz.read(cov_entry)
                form_bytes = rz.read(form_entry)
                cov = pd.read_csv(io.BytesIO(cov_bytes), header=None).values
                form = pd.read_csv(io.BytesIO(form_bytes), header=None).values
                return cov, form
        except Exception as exc:
            if attempt == 2:
                print(f"[RemoteZip Error] PID {pid} failed after 3 attempts: {exc}")
                return None, None
            time.sleep(1.5 * (attempt + 1))

    return None, None


def process_session(pid: str, data_root: Path, session: Session) -> bool:
    """Extract and save acoustic features for one session."""
    p_dir = data_root / f"{pid}_P"
    out_parquet = p_dir / f"{pid}_ACOUSTIC_WINDOWS.parquet"

    if out_parquet.exists() and out_parquet.stat().st_size > 100:
        return True

    raw_part_turns = get_participant_turns_from_session(session)
    if not raw_part_turns:
        return False

    cov, form = get_session_acoustic_arrays(pid)
    if cov is None or form is None:
        return False

    n_frames = min(len(cov), len(form))
    window_size = 5
    n_windows = int(np.ceil(len(raw_part_turns) / window_size))

    rows: list[dict[str, Any]] = []
    for w_idx in range(n_windows):
        st_turn = w_idx * window_size
        en_turn = min((w_idx + 1) * window_size, len(raw_part_turns))
        chunk = raw_part_turns[st_turn:en_turn]

        cov_frames: list[np.ndarray] = []
        form_frames: list[np.ndarray] = []

        for t in chunk:
            if t.in_interruption_span or t.has_scrubbed:
                continue
            st = int(np.floor(t.start_s * 100))
            en = int(np.ceil(t.end_s * 100))
            if st < n_frames:
                en = min(en, n_frames)
                if en > st:
                    cov_frames.append(cov[st:en])
                    form_frames.append(form[st:en])

        if cov_frames:
            cov_win = np.vstack(cov_frames)
            form_win = np.vstack(form_frames)
        else:
            cov_win = np.empty((0, 74))
            form_win = np.empty((0, 5))

        feats = compute_window_acoustics(cov_win, form_win)
        rows.append({"participant_id": str(pid), "window_idx": w_idx, **feats})

    df_out = pd.DataFrame(rows)
    p_dir.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_parquet, index=False)
    return True


def main():
    data_root = Path("data/daic_woz")
    loader = DAICLoader(data_root)

    print("Loading all sessions from disk...")
    sessions_map = {str(s.participant_id): s for s in loader.iter_sessions()}
    pids = sorted(sessions_map.keys(), key=lambda x: int(x) if x.isdigit() else 9999)
    print(f"Loaded {len(pids)} sessions from {data_root}.")

    already_done = [
        pid for pid in pids if (data_root / f"{pid}_P" / f"{pid}_ACOUSTIC_WINDOWS.parquet").exists()
    ]
    print(f"{len(already_done)}/{len(pids)} sessions already have acoustic parquets.")

    remaining = [pid for pid in pids if pid not in already_done]
    print(f"Extracting acoustics for {len(remaining)} remaining sessions with 8 workers...")

    success_count = len(already_done)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_pid = {
            executor.submit(process_session, pid, data_root, sessions_map[pid]): pid
            for pid in remaining
        }
        for future in concurrent.futures.as_completed(future_to_pid):
            pid = future_to_pid[future]
            try:
                ok = future.result()
                if ok:
                    success_count += 1
                    if success_count % 10 == 0 or success_count == len(pids):
                        print(f"[{success_count}/{len(pids)}] Completed acoustic extraction.")
                else:
                    print(f"[WARN] Failed to extract acoustics for PID {pid}")
            except Exception as exc:
                print(f"[ERROR] Exception on PID {pid}: {exc}")

    print(f"\nFinal status: {success_count}/{len(pids)} sessions extracted successfully.")


if __name__ == "__main__":
    main()

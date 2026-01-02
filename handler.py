import os
import sys
import shutil
import subprocess
import uuid
from pathlib import Path

import requests
import runpod

WORKDIR = Path("/workspace")
WAV2LIP_DIR = WORKDIR / "Wav2Lip"
INPUT_DIR = WORKDIR / "input"
OUTPUT_DIR = WORKDIR / "output"

INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def diag():
    # This prints into RunPod logs so you can prove which image/code is running.
    print("=== DIAG START ===")
    print("DIAG sys.executable:", sys.executable)
    print("DIAG which python:", shutil.which("python"))
    print("DIAG which python3:", shutil.which("python3"))
    try:
        out = subprocess.check_output(
            ["bash", "-lc", "ls -l /usr/bin/python* || true; python3 --version || true; python --version || true"],
            text=True
        )
        print("DIAG bash:\n", out)
    except Exception as e:
        print("DIAG bash error:", repr(e))
    print("DIAG handler path:", __file__)
    print("DIAG cwd:", os.getcwd())
    print("=== DIAG END ===")

diag()

def download(url: str, out_path: Path):
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

def get_audio_duration_seconds(audio_path: Path) -> float:
    """
    Uses ffprobe to get audio duration so we can make the face video match the audio length.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path)
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    try:
        return max(0.1, float(out))
    except Exception:
        # fallback
        return 30.0

def handler(job):
    inp = job.get("input", {})

    image_url = inp.get("image_url")
    audio_url = inp.get("audio_url")
    fps = int(inp.get("fps", 25))
    pads = inp.get("pads", [0, 10, 0, 0])
    resize_factor = int(inp.get("resize_factor", 1))

    if not image_url or not audio_url:
        return {"error": "Missing image_url or audio_url"}

    job_id = str(uuid.uuid4())[:8]
    face_path = INPUT_DIR / f"face_{job_id}.jpg"
    audio_path = INPUT_DIR / f"audio_{job_id}.wav"
    temp_face_video = INPUT_DIR / f"face_{job_id}.mp4"
    out_path = OUTPUT_DIR / f"out_{job_id}.mp4"

    # Download inputs
    download(image_url, face_path)
    download(audio_url, audio_path)

    # Make the still image into a video matching the audio length
    duration = get_audio_duration_seconds(audio_path)

    subprocess.check_call([
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", str(face_path),
        "-t", str(duration),
        "-r", str(fps),
        "-vf", "format=yuv420p",
        str(temp_face_video)
    ])

    # Run inference using the current interpreter (never calls "python")
    inference_path = WAV2LIP_DIR / "inference.py"
    if not inference_path.exists():
        return {"error": f"inference.py not found at {inference_path}"}

    subprocess.check_call([
        sys.executable, str(inference_path),
        "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
        "--face", str(temp_face_video),
        "--audio", str(audio_path),
        "--outfile", str(out_path),
        "--pads", str(pads[0]), str(pads[1]), str(pads[2]), str(pads[3]),
        "--resize_factor", str(resize_factor)
    ], cwd=str(WAV2LIP_DIR))

    return {"video_path": str(out_path), "job_id": job_id}

runpod.serverless.start({"handler": handler})

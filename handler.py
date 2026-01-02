import os
import subprocess
import uuid
from pathlib import Path
import requests
import runpod

WORKDIR = Path("/workspace")
INPUT_DIR = WORKDIR / "input"
OUTPUT_DIR = WORKDIR / "output"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download(url: str, out_path: Path):
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

def handler(job):
    inp = job["input"]

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
    out_path = OUTPUT_DIR / f"out_{job_id}.mp4"

    download(image_url, face_path)
    download(audio_url, audio_path)

    temp_face_video = INPUT_DIR / f"face_{job_id}.mp4"
    subprocess.check_call([
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", str(face_path),
        "-t", "30",
        "-r", str(fps),
        "-vf", "format=yuv420p",
        str(temp_face_video)
    ])

    subprocess.check_call([
    "python3", "inference.py",
        "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
        "--face", str(temp_face_video),
        "--audio", str(audio_path),
        "--outfile", str(out_path),
        "--pads", str(pads[0]), str(pads[1]), str(pads[2]), str(pads[3]),
        "--resize_factor", str(resize_factor)
    ])

    return {"video_path": str(out_path)}

runpod.serverless.start({"handler": handler})

#!/usr/bin/env python3
"""
Train a custom wake word model using the openWakeWord framework.

openWakeWord trains a small DNN on top of pre-computed audio embeddings
(ACAV100M). The final model is ~860 KB TFLite and runs in real time on CPU.

Usage:
    python3 train_wakeword.py --config example_ru_jarvis.yaml [--phase PHASE]

Phases (run in order):
    setup      Write the generate_samples wrapper (needed by train.py)
    voices     Download Piper ONNX voice models from HuggingFace
    features   Download ACAV100M + validation feature files (~7 GB total)
    background Download MIT RIRs, AudioSet, and FMA background audio (~5 GB)
    speech     Extract real speech features as a dedicated negative class
    generate   Generate TTS clips via openWakeWord train.py --generate_clips
    augment    Augment clips with room impulse responses + background noise
    train      Train the DNN via openWakeWord train.py --train_model
    export     Convert ONNX → TFLite → copy <model_name>.tflite to project root

Run setup.sh first to create the Python virtualenv and download dependencies.

Output files (created at project root after export phase):
    <model_name>.tflite    — TFLite model ready for use with openWakeWord
    <model_name>.onnx      — ONNX model (if TFLite conversion fails)
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# CONFIG LOADING
# ===========================================================================

def load_config(config_path: str) -> dict:
    """Load and validate a YAML config file, returning a flat config dict.

    Required fields: model_name, target_phrases, voices.primary
    All other fields have sensible defaults.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw:
        raise ValueError(f"Config file is empty: {config_path}")

    # --- Required fields ---
    if "model_name" not in raw:
        raise ValueError("Config missing required field: model_name")
    if "target_phrases" not in raw or not raw["target_phrases"]:
        raise ValueError("Config missing required field: target_phrases")
    if "voices" not in raw or "primary" not in raw.get("voices", {}):
        raise ValueError("Config missing required field: voices.primary")

    primary = raw["voices"]["primary"]
    if "base_url" not in primary or "models" not in primary:
        raise ValueError("voices.primary must have 'base_url' and 'models'")

    # --- Build flat config dict with defaults ---
    training = raw.get("training", {})
    cfg = {
        "model_name":      raw["model_name"],
        "target_phrases":  raw["target_phrases"],
        "n_samples":       training.get("n_samples", 50_000),
        "n_samples_val":   training.get("n_samples_val", 10_000),
        "training_steps":  training.get("steps", 500_000),
        "max_neg_weight":  training.get("max_neg_weight", 3_000),
        "layer_size":      training.get("layer_size", 128),

        # Voices
        "primary_voices_base_url": primary["base_url"],
        "primary_voices":          primary["models"],

        # Secondary voices (optional — used for cross-language negatives)
        "secondary_voices_base_url": raw["voices"].get("secondary", {}).get("base_url", ""),
        "secondary_voices":          raw["voices"].get("secondary", {}).get("models", {}),

        # Adversarial negative phrases (optional but strongly recommended)
        "negative_phrases": raw.get("negative_phrases", []),
    }

    # Russian speech negatives (optional — requires HuggingFace dataset)
    # "speech_negatives" is the canonical key; "russian_speech_negatives" is
    # accepted for backward compatibility with older configs.
    ru_neg = raw.get("speech_negatives", raw.get("russian_speech_negatives", {}))
    cfg["ru_speech_enabled"]    = ru_neg.get("enabled", False)
    cfg["ru_speech_dataset"]    = ru_neg.get("dataset", "bond005/sberdevices_golos_10h_crowd")
    cfg["ru_speech_split"]      = ru_neg.get("split", "train")
    cfg["ru_speech_max_chunks"] = ru_neg.get("max_chunks", 5000)
    cfg["ru_speech_batch_n"]    = ru_neg.get("batch_n", 200)

    return cfg


# ===========================================================================
# CONFIGURATION — populated by load_config() in main()
# ===========================================================================

TARGET_PHRASES: list[str] = []
MODEL_NAME: str = ""
N_SAMPLES: int = 50_000
N_SAMPLES_VAL: int = 10_000
TRAINING_STEPS: int = 500_000
MAX_NEG_WEIGHT: int = 3_000
LAYER_SIZE: int = 128

# Voice configs
PRIMARY_VOICES_BASE_URL: str = ""
PRIMARY_VOICES: dict[str, str] = {}
SECONDARY_VOICES_BASE_URL: str = ""
SECONDARY_VOICES: dict[str, str] = {}

NEGATIVE_PHRASES: list[str] = []

# Russian speech negatives
RU_SPEECH_ENABLED: bool = False
RU_SPEECH_DATASET: str = "bond005/sberdevices_golos_10h_crowd"
RU_SPEECH_SPLIT: str = "train"
RU_SPEECH_MAX_CHUNKS: int = 5000
RU_SPEECH_BATCH_N: int = 200

# ---------------------------------------------------------------------------
# Workspace layout
# ---------------------------------------------------------------------------
WORK_DIR     = Path("training")
REPOS_DIR    = WORK_DIR / "repos"
OWW_DIR      = REPOS_DIR / "openwakeword"
WRAPPER_DIR  = WORK_DIR / "piper_wrapper"   # piper_sample_generator_path for train.py
MODELS_DIR   = WORK_DIR / "piper_models"
RIRS_DIR     = WORK_DIR / "mit_rirs"
BG_DIR       = WORK_DIR / "background"
FEATURES_DIR = WORK_DIR / "features"
RU_SPEECH_DIR = WORK_DIR / "ru_speech"
OUTPUT_DIR   = WORK_DIR / "output"

# Piper standalone binary (downloaded by setup.sh).
# The binary bundles espeak-ng-data and shared libs — no Python TTS package needed.
PIPER_BIN_DIR = WORK_DIR / "piper_binary" / "piper"   # directory extracted from tar
PIPER_BIN     = PIPER_BIN_DIR / "piper"               # the executable itself
PIPER_ESPEAK  = PIPER_BIN_DIR / "espeak-ng-data"      # bundled espeak data

TRAIN_SCRIPT = OWW_DIR / "openwakeword" / "train.py"

# Absolute path to the Python interpreter running this script.
# Using sys.executable ensures subprocess calls use the venv Python,
# not whatever `python3` resolves to in the shell (system Python on Debian/Ubuntu
# may be "externally managed" and refuse pip installs).
PYTHON = sys.executable


def _apply_config(cfg: dict):
    """Copy config dict values into module-level constants."""
    global TARGET_PHRASES, MODEL_NAME, N_SAMPLES, N_SAMPLES_VAL
    global TRAINING_STEPS, MAX_NEG_WEIGHT, LAYER_SIZE
    global PRIMARY_VOICES_BASE_URL, PRIMARY_VOICES
    global SECONDARY_VOICES_BASE_URL, SECONDARY_VOICES
    global NEGATIVE_PHRASES
    global RU_SPEECH_ENABLED, RU_SPEECH_DATASET, RU_SPEECH_SPLIT
    global RU_SPEECH_MAX_CHUNKS, RU_SPEECH_BATCH_N

    MODEL_NAME       = cfg["model_name"]
    TARGET_PHRASES   = cfg["target_phrases"]
    N_SAMPLES        = cfg["n_samples"]
    N_SAMPLES_VAL    = cfg["n_samples_val"]
    TRAINING_STEPS   = cfg["training_steps"]
    MAX_NEG_WEIGHT   = cfg["max_neg_weight"]
    LAYER_SIZE       = cfg["layer_size"]

    PRIMARY_VOICES_BASE_URL   = cfg["primary_voices_base_url"]
    PRIMARY_VOICES            = cfg["primary_voices"]
    SECONDARY_VOICES_BASE_URL = cfg["secondary_voices_base_url"]
    SECONDARY_VOICES          = cfg["secondary_voices"]

    NEGATIVE_PHRASES = cfg["negative_phrases"]

    RU_SPEECH_ENABLED    = cfg.get("ru_speech_enabled", False)
    RU_SPEECH_DATASET    = cfg.get("ru_speech_dataset", "bond005/sberdevices_golos_10h_crowd")
    RU_SPEECH_SPLIT      = cfg.get("ru_speech_split", "train")
    RU_SPEECH_MAX_CHUNKS = cfg.get("ru_speech_max_chunks", 5000)
    RU_SPEECH_BATCH_N    = cfg.get("ru_speech_batch_n", 200)


# ===========================================================================
# UTILITIES
# ===========================================================================

def run(cmd: str, **kw):
    """Execute a shell command, streaming output to the terminal."""
    log.info("$ %s", cmd)
    subprocess.run(cmd, shell=True, check=True, **kw)


def wget(url: str, dest: Path):
    """Download url → dest, skipping if the destination already looks complete."""
    if dest.exists() and dest.stat().st_size > 4096:
        log.info("  skip (exists): %s", dest.name)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(f'wget -q --show-progress -O "{dest}" "{url}"')


def count_wav(directory: Path) -> int:
    return len(list(directory.glob("*.wav")))


def _yaml_path() -> Path:
    """Path for the openWakeWord train.py YAML config (generated at runtime)."""
    return WORK_DIR / f"{MODEL_NAME}.yaml"


# ===========================================================================
# PHASE 0: Write generate_samples wrapper
# ===========================================================================
# openWakeWord's train.py does:
#     sys.path.insert(0, config["piper_sample_generator_path"])
#     from generate_samples import generate_samples
#
# The upstream piper-sample-generator's generate_samples() only handles .pt
# (PyTorch) models and requires a model= argument that train.py never passes.
# We solve both issues by writing our own generate_samples.py wrapper that:
#   - Accepts the exact same signature train.py expects
#   - Internally synthesises WAV clips using the piper standalone binary
#   - Filters empty texts that generate_adversarial_texts sometimes returns
#
# The piper standalone binary is used instead of piper-tts (Python package)
# because piper-phonemize has no Python 3.12 wheel.  The binary is
# self-contained — it bundles its own espeak-ng-data and shared libs.
# ===========================================================================

WRAPPER_TEMPLATE = textwrap.dedent("""\
    # Auto-generated by train_wakeword.py — do not edit manually.
    # Uses piper standalone binary for TTS (no piper-tts / piper-phonemize needed).
    import os, subprocess, uuid, random, logging
    import numpy as np
    import scipy.io.wavfile
    import scipy.signal
    from pathlib import Path
    from typing import Union, List, Optional, Iterable, Tuple
    from tqdm import tqdm

    _TARGET_SR = 16000  # openWakeWord expects 16 kHz

    _LOG = logging.getLogger(__name__)

    # Paths resolved at wrapper-generation time
    _PIPER_BIN   = r"{piper_bin}"       # path to piper executable
    _PIPER_LIBS  = r"{piper_libs}"      # dir with bundled .so files
    _ESPEAK_DATA = r"{espeak_data}"     # dir with espeak-ng-data

    # ONNX voice model paths (embedded at generation time)
    _VOICE_MODELS = {voice_models!r}

    # Fallback phrases when adversarial text generation returns empty list
    _FALLBACK_NEG = {fallback_negatives!r}

    # Expose the bundled shared libs so the binary finds them at runtime
    _ld = os.environ.get("LD_LIBRARY_PATH", "")
    if _PIPER_LIBS and _PIPER_LIBS not in _ld:
        os.environ["LD_LIBRARY_PATH"] = (_PIPER_LIBS + ":" + _ld) if _ld else _PIPER_LIBS


    def _synth_one(phrase: str, model_onnx: str, out_path: Path,
                   length_scale: float, noise_scale: float, noise_w: float) -> bool:
        \"\"\"Call piper binary to synthesise a single WAV file at 16 kHz. Returns True on success.\"\"\"
        env = dict(os.environ)
        if _ESPEAK_DATA:
            env["ESPEAK_DATA_PATH"] = _ESPEAK_DATA
        result = subprocess.run(
            [_PIPER_BIN, "--model", model_onnx, "--output_file", str(out_path),
             "--length_scale", str(length_scale),
             "--noise_scale",  str(noise_scale),
             "--noise_w",      str(noise_w)],
            input=phrase, text=True, capture_output=True, env=env,
        )
        if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size <= 44:
            return False

        # Resample to 16 kHz if the voice model outputs a different sample rate
        sr, data = scipy.io.wavfile.read(out_path)
        if sr != _TARGET_SR:
            n_samples = int(len(data) * _TARGET_SR / sr)
            data = scipy.signal.resample(data, n_samples).astype(data.dtype)
            scipy.io.wavfile.write(str(out_path), _TARGET_SR, data)

        return True


    def generate_samples(
        text: Union[List[str], str],
        output_dir: Union[str, Path],
        model: Optional[Union[str, List[str]]] = None,
        max_samples: Optional[int] = None,
        file_names: Optional[Iterable[str]] = None,
        # .pt-model kwargs — silently ignored (not applicable to ONNX voices)
        batch_size=None,
        slerp_weights=None,
        max_speakers=None,
        auto_reduce_batch_size=None,
        # TTS variation parameters
        length_scales: Tuple[float, ...] = (0.75, 1.0, 1.1, 1.25),
        noise_scales:  Tuple[float, ...] = (0.667, 0.8, 0.98),
        noise_scale_ws: Tuple[float, ...] = (0.8, 0.9, 0.98),
        **kwargs,
    ) -> int:
        \"\"\"Synthesise clips using the piper binary with ONNX voice models.

        Drop-in replacement for piper-sample-generator's generate_samples().
        \"\"\"
        voices = model or _VOICE_MODELS
        if not voices:
            raise RuntimeError(
                "No voice models found. Run: python3 train_wakeword.py --config <config.yaml> --phase voices"
            )
        if isinstance(voices, str):
            voices = [voices]

        # Filter blank texts (generate_adversarial_texts may produce some for non-English)
        if isinstance(text, (list, tuple)):
            text = [t for t in text if t and t.strip()]
        else:
            text = [text] if text and text.strip() else []
        if not text:
            _LOG.warning("Empty text list — substituting fallback negatives")
            text = _FALLBACK_NEG

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build a shuffled list of all (phrase, voice, ls, ns, nw) combinations.
        # Cycling through it gives uniform coverage of all acoustic variations.
        all_combos = [
            (t, v, ls, ns, nw)
            for t  in text
            for v  in voices
            for ls in length_scales
            for ns in noise_scales
            for nw in noise_scale_ws
        ]
        random.shuffle(all_combos)
        n_combos = len(all_combos)

        names_it = iter(file_names) if file_names is not None else None
        target   = max_samples or n_combos
        generated, idx = 0, 0

        _LOG.info(
            "Generating %d clips  (%d phrase(s) x %d voice(s) x %d param combos)",
            target, len(text), len(voices),
            len(length_scales) * len(noise_scales) * len(noise_scale_ws),
        )

        pbar = tqdm(total=target, desc="Generating clips", unit="clip")
        while generated < target:
            phrase, voice, ls, ns, nw = all_combos[idx % n_combos]
            idx += 1

            fname = ((next(names_it, None) if names_it is not None else None)
                    or uuid.uuid4().hex + ".wav")
            out   = output_dir / fname

            if _synth_one(phrase, voice, out, ls, ns, nw):
                generated += 1
                pbar.update(1)
            else:
                _LOG.debug("Synthesis failed  phrase=%r  voice=%s", phrase, Path(voice).stem)

        pbar.close()
        _LOG.info("Done: %d clips -> %s", generated, output_dir)
        return generated
""")


def phase_setup():
    log.info("=" * 60)
    log.info("Phase: setup — writing generate_samples wrapper")
    log.info("=" * 60)

    # Verify the piper binary exists (downloaded by setup.sh)
    if not PIPER_BIN.exists():
        raise FileNotFoundError(
            f"Piper binary not found: {PIPER_BIN}\n"
            "Run setup.sh first to download it."
        )
    log.info("  piper binary: %s", PIPER_BIN)

    WRAPPER_DIR.mkdir(parents=True, exist_ok=True)

    # Embed voice model paths into the wrapper.
    # These may not exist yet (they're downloaded in phase_voices), but the
    # paths are embedded now so the wrapper doesn't need to look them up at
    # generation time.
    voice_paths = [
        str((MODELS_DIR / f"{name}.onnx").resolve())
        for name in PRIMARY_VOICES
    ]

    wrapper_src = WRAPPER_TEMPLATE.format(
        piper_bin=str(PIPER_BIN.resolve()),
        piper_libs=str(PIPER_BIN_DIR.resolve()),
        espeak_data=str(PIPER_ESPEAK.resolve()),
        voice_models=voice_paths,
        fallback_negatives=NEGATIVE_PHRASES[:10],
    )

    wrapper_file = WRAPPER_DIR / "generate_samples.py"
    wrapper_file.write_text(wrapper_src, encoding="utf-8")
    log.info("  Wrapper written: %s", wrapper_file)


# ===========================================================================
# PHASE 1: Download Piper ONNX voice models
# ===========================================================================

def phase_voices() -> list[str]:
    log.info("=" * 60)
    log.info("Phase: voices — downloading Piper voices")
    log.info("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    available = []

    # Download primary voices (used for positive clips and same-language negatives)
    for name, path_suffix in PRIMARY_VOICES.items():
        onnx = MODELS_DIR / f"{name}.onnx"
        json_f = MODELS_DIR / f"{name}.onnx.json"
        base = f"{PRIMARY_VOICES_BASE_URL}/{path_suffix}/{name}"
        try:
            wget(f"{base}.onnx", onnx)
            wget(f"{base}.onnx.json", json_f)
            available.append(str(onnx.resolve()))
            log.info("  ✓ %s", name)
        except subprocess.CalledProcessError as exc:
            log.warning("  Failed to download %s: %s", name, exc)

    if not available:
        raise RuntimeError(
            "No primary voices downloaded. Check your internet connection and voice URLs."
        )
    log.info("  %d primary voice(s) available", len(available))

    # Download secondary voices (used for cross-language negatives only)
    if SECONDARY_VOICES and SECONDARY_VOICES_BASE_URL:
        sec_available = []
        for name, path_suffix in SECONDARY_VOICES.items():
            onnx = MODELS_DIR / f"{name}.onnx"
            json_f = MODELS_DIR / f"{name}.onnx.json"
            base = f"{SECONDARY_VOICES_BASE_URL}/{path_suffix}/{name}"
            try:
                wget(f"{base}.onnx", onnx)
                wget(f"{base}.onnx.json", json_f)
                sec_available.append(str(onnx.resolve()))
                log.info("  ✓ %s", name)
            except subprocess.CalledProcessError as exc:
                log.warning("  Failed to download %s: %s", name, exc)

        if sec_available:
            log.info("  %d secondary voice(s) available", len(sec_available))
        else:
            log.warning("  No secondary voices downloaded — cross-language negatives will be skipped")
    else:
        log.info("  No secondary voices configured — skipping")

    return available


# ===========================================================================
# PHASE 2: Download pre-computed openWakeWord feature files
# ===========================================================================

def phase_features():
    """Download ACAV100M embeddings and validation feature files from HuggingFace.

    These are pre-computed audio embeddings (~7 GB total):
      openwakeword_features_ACAV100M_2000_hrs_16bit.npy — training negatives
      validation_set_features.npy                       — FP validation set
    """
    log.info("=" * 60)
    log.info("Phase: features — downloading pre-computed feature files (~7 GB)")
    log.info("=" * 60)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    files = [
        "openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
        "validation_set_features.npy",
    ]
    for fname in files:
        dest = FEATURES_DIR / fname
        if dest.exists() and dest.stat().st_size > 1_000_000:
            log.info("  skip (exists): %s", fname)
            continue
        log.info("  Downloading %s ...", fname)
        hf_hub_download(
            repo_id="davidscripka/openwakeword_features",
            filename=fname,
            repo_type="dataset",
            local_dir=str(FEATURES_DIR),
        )
        log.info("  ✓ %s", fname)


# ===========================================================================
# PHASE 3: Download background audio
# ===========================================================================

def phase_background():
    """Download MIT Room Impulse Responses, AudioSet, and FMA background audio.

    Used by openWakeWord's augmentation step to simulate real-world acoustic
    conditions (room reverberation, background noise, music).
    """
    log.info("=" * 60)
    log.info("Phase: background — downloading background audio")
    log.info("=" * 60)

    RIRS_DIR.mkdir(parents=True, exist_ok=True)
    BG_DIR.mkdir(parents=True, exist_ok=True)

    _download_rirs()
    _download_audioset()
    _download_fma()


def _audio_to_wav16k(array: np.ndarray, sr: int) -> np.ndarray:
    """Resample to 16 kHz and convert to int16."""
    if sr != 16000:
        try:
            import librosa
            array = librosa.resample(array.astype(np.float32), orig_sr=sr, target_sr=16000)
        except ImportError:
            # linear interpolation fallback when librosa is unavailable
            ratio = 16000 / sr
            new_len = int(len(array) * ratio)
            array = np.interp(
                np.linspace(0, len(array) - 1, new_len),
                np.arange(len(array)),
                array,
            )
    return (array * 32767).clip(-32768, 32767).astype(np.int16)


def _download_rirs():
    """Download MIT Environmental Impulse Responses from HuggingFace."""
    rir_wav_dir = RIRS_DIR / "16khz"
    if rir_wav_dir.exists() and count_wav(rir_wav_dir) > 10:
        log.info("  MIT RIRs already downloaded (%d files)", count_wav(rir_wav_dir))
        return

    log.info("  Downloading MIT Room Impulse Responses via snapshot_download ...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="davidscripka/MIT_environmental_impulse_responses",
        repo_type="dataset",
        local_dir=str(RIRS_DIR),
        ignore_patterns=["*.md", ".gitattributes"],
    )
    n = count_wav(rir_wav_dir)
    if n == 0:
        log.warning("  No WAV files found after download; augmentation may be affected")
    else:
        log.info("  ✓ MIT RIRs: %d files", n)


def _download_audioset():
    """Download AudioSet background noise clips from HuggingFace (streaming, up to 2000)."""
    audioset_dir = BG_DIR / "audioset"
    if audioset_dir.exists() and count_wav(audioset_dir) > 100:
        log.info("  AudioSet already downloaded (%d files)", count_wav(audioset_dir))
        return

    audioset_dir.mkdir(parents=True, exist_ok=True)
    log.info("  Downloading AudioSet background noise subset (up to 2000 clips)...")
    try:
        from datasets import load_dataset
        import scipy.io.wavfile as wavfile

        ds = load_dataset(
            "agkphysics/AudioSet",
            "balanced",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        count, limit = 0, 2000
        for item in ds:
            if count >= limit:
                break
            try:
                audio = item["audio"]
                arr = _audio_to_wav16k(np.array(audio["array"]), audio["sampling_rate"])
                wavfile.write(str(audioset_dir / f"as_{count:05d}.wav"), 16000, arr)
                count += 1
            except Exception as exc:
                log.debug("  Skipping AudioSet item: %s", exc)
        log.info("  ✓ AudioSet: %d files", count)
    except Exception as exc:
        log.warning("  AudioSet download failed (%s) — creating noise stand-in", exc)
        _make_noise_background(audioset_dir, n=50)


def _download_fma():
    """Download FMA music clips from HuggingFace (streaming, up to 1000)."""
    fma_dir = BG_DIR / "fma"
    if fma_dir.exists() and count_wav(fma_dir) > 100:
        log.info("  FMA already downloaded (%d files)", count_wav(fma_dir))
        return

    fma_dir.mkdir(parents=True, exist_ok=True)
    log.info("  Downloading FMA music subset (up to 1000 clips)...")
    try:
        from datasets import load_dataset
        import scipy.io.wavfile as wavfile

        ds = load_dataset(
            "rudraml/fma",
            name="small",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        count, limit = 0, 1000
        for item in ds:
            if count >= limit:
                break
            try:
                audio = item["audio"]
                arr = _audio_to_wav16k(np.array(audio["array"]), audio["sampling_rate"])
                wavfile.write(str(fma_dir / f"fma_{count:05d}.wav"), 16000, arr)
                count += 1
            except Exception as exc:
                log.debug("  Skipping FMA item: %s", exc)
        log.info("  ✓ FMA: %d files", count)
    except Exception as exc:
        log.warning("  FMA download failed (%s) — creating noise stand-in", exc)
        _make_noise_background(fma_dir, n=50)


def _make_noise_background(directory: Path, n: int = 50):
    """Create placeholder white-noise WAV files when a dataset download fails."""
    import scipy.io.wavfile as wavfile
    for i in range(n):
        arr = (np.random.randn(16000 * 10) * 150).astype(np.int16)  # 10 s
        wavfile.write(str(directory / f"noise_{i:04d}.wav"), 16000, arr)
    log.info("  Created %d placeholder noise files in %s", n, directory)


# ===========================================================================
# PHASE: ru_speech — Russian speech features as a negative class
# ===========================================================================

def phase_ru_speech():
    """Download Russian speech, chunk it, and extract openWakeWord features.

    Creates training/ru_speech/ru_speech_features.npy — a dedicated negative
    class that teaches the model to reject general Russian conversation.
    Only runs if speech_negatives.enabled is true in the config.
    """
    log.info("=" * 60)
    log.info("Phase: ru_speech — Russian speech negative features")
    log.info("=" * 60)

    if not RU_SPEECH_ENABLED:
        log.info("  speech_negatives.enabled is false — skipping")
        return

    ru_npy = RU_SPEECH_DIR / "ru_speech_features.npy"
    if ru_npy.exists() and ru_npy.stat().st_size > 1_000_000:
        log.info("  Russian speech features already exist (%s, %.1f MB) — skipping",
                 ru_npy, ru_npy.stat().st_size / 1e6)
        return

    try:
        from datasets import load_dataset

        _ensure_oww_feature_models()

        # Use openWakeWord's AudioFeatures for the correct mel→embedding pipeline
        sys.path.insert(0, str(OWW_DIR))
        from openwakeword.utils import AudioFeatures
        audio_features = AudioFeatures(device="cpu")

        log.info("  Streaming dataset: %s (split=%s)", RU_SPEECH_DATASET, RU_SPEECH_SPLIT)
        ds = load_dataset(RU_SPEECH_DATASET, split=RU_SPEECH_SPLIT, streaming=True)

        # Collect 2-second chunks of 16kHz int16 audio
        CHUNK_SAMPLES = 32000  # 2 seconds at 16kHz
        chunks = []
        total_items = 0
        for item in ds:
            if len(chunks) >= RU_SPEECH_MAX_CHUNKS:
                break
            try:
                audio = item["audio"]
                arr = _audio_to_wav16k(np.array(audio["array"]), audio["sampling_rate"])
                for start in range(0, len(arr) - CHUNK_SAMPLES + 1, CHUNK_SAMPLES):
                    chunks.append(arr[start:start + CHUNK_SAMPLES])
                    if len(chunks) >= RU_SPEECH_MAX_CHUNKS:
                        break
                total_items += 1
            except Exception as exc:
                log.debug("  Skipping audio item: %s", exc)
            if total_items % 200 == 0 and total_items > 0:
                log.info("  Processed %d items → %d chunks so far", total_items, len(chunks))

        log.info("  Collected %d chunks from %d audio items", len(chunks), total_items)
        if not chunks:
            log.warning("  No chunks collected — skipping Russian speech features")
            return

        # Extract openWakeWord embeddings from the collected audio chunks
        clip_array = np.stack(chunks)  # (N, 32000) int16
        log.info("  Extracting features from %d chunks (batch_size=64)...", len(chunks))
        features = audio_features.embed_clips(clip_array, batch_size=64)
        # features shape: (N, 16, 96) — convert to float16 to match ACAV100M format
        features = features.astype(np.float16)

        RU_SPEECH_DIR.mkdir(parents=True, exist_ok=True)
        np.save(str(ru_npy), features)
        log.info("  ✓ Saved Russian speech features: %s (%.1f MB, %d vectors)",
                 ru_npy, ru_npy.stat().st_size / 1e6, features.shape[0])

    except Exception as exc:
        log.warning("  Russian speech feature extraction failed: %s", exc)
        log.warning("  Training will proceed without the russian_speech negative class")


# ===========================================================================
# PHASE 4: Write openWakeWord train.py YAML config
# ===========================================================================

def write_yaml():
    """Write training/<model_name>.yaml for openWakeWord train.py."""
    import yaml as _yaml

    # Verify required feature files exist before writing the config
    for fname in ("openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
                  "validation_set_features.npy"):
        p = FEATURES_DIR / fname
        if not p.exists():
            raise FileNotFoundError(
                f"Required feature file not found: {p}\n"
                f"Run: python3 train_wakeword.py --config <config.yaml> --phase features"
            )

    rir_wav_dir = RIRS_DIR / "16khz"
    if not rir_wav_dir.exists() or count_wav(rir_wav_dir) == 0:
        raise FileNotFoundError(
            f"No RIR files found in {rir_wav_dir}\n"
            f"Run: python3 train_wakeword.py --config <config.yaml> --phase background"
        )

    config = {
        "model_name": MODEL_NAME,
        "target_phrase": TARGET_PHRASES,
        # Provide rich negatives as a safety net in case
        # generate_adversarial_texts returns empty for the target language.
        "custom_negative_phrases": NEGATIVE_PHRASES,
        "n_samples":           N_SAMPLES,
        "n_samples_val":       N_SAMPLES_VAL,
        # tts_batch_size is ignored by our wrapper (ONNX is sequential)
        # but must be present in the YAML for train.py.
        "tts_batch_size": 1,
        "augmentation_batch_size": 16,
        # Points to our wrapper, not the real piper-sample-generator.
        "piper_sample_generator_path": str(WRAPPER_DIR.resolve()),
        "output_dir": str(OUTPUT_DIR.resolve()),
        "rir_paths": [str(rir_wav_dir.resolve())],
        "background_paths": [
            str((BG_DIR / "audioset").resolve()),
            str((BG_DIR / "fma").resolve()),
        ],
        "background_paths_duplication_rate": [1, 1],
        "false_positive_validation_data_path": str(
            (FEATURES_DIR / "validation_set_features.npy").resolve()
        ),
        "augmentation_rounds": 1,
        "feature_data_files": {
            "ACAV100M_sample": str(
                (FEATURES_DIR / "openwakeword_features_ACAV100M_2000_hrs_16bit.npy").resolve()
            ),
        },
        "batch_n_per_class": {
            "ACAV100M_sample":      1024,
            "adversarial_negative": 50,
            "positive":             50,
        },
    }

    # Add Russian speech negative class if enabled and features file exists
    ru_npy = RU_SPEECH_DIR / "ru_speech_features.npy"
    if RU_SPEECH_ENABLED and ru_npy.exists():
        config["feature_data_files"]["russian_speech"] = str(ru_npy.resolve())
        config["batch_n_per_class"]["russian_speech"] = RU_SPEECH_BATCH_N
        log.info("  Including russian_speech negative class (batch_n=%d)", RU_SPEECH_BATCH_N)

    config.update({
        "model_type":  "dnn",
        "layer_size":  LAYER_SIZE,
        "steps":       TRAINING_STEPS,
        "max_negative_weight":             MAX_NEG_WEIGHT,
        "target_false_positives_per_hour": 0.2,
    })

    yaml_path = _yaml_path()
    with open(yaml_path, "w", encoding="utf-8") as f:
        _yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    log.info("  YAML written: %s", yaml_path)
    return yaml_path


# ===========================================================================
# PHASE 5: Generate TTS clips (via openWakeWord train.py --generate_clips)
# ===========================================================================

def _is_latin(text: str) -> bool:
    """Return True if text is predominantly Latin script."""
    latin = sum(1 for c in text if c.isalpha() and ord(c) < 256)
    total = sum(1 for c in text if c.isalpha())
    return total > 0 and latin / total > 0.5


def _generate_secondary_negatives():
    """Synthesise secondary-language negative phrases with secondary voice models.

    Detects which negative phrases use a different script than the target
    phrases and synthesises them with the secondary voice set. Generates
    ~2000 clips split 80/20 into negative_train / negative_test.

    This helps the model reject speech from non-primary languages (e.g. English
    words on TV/YouTube that sound similar to a Russian wake word).
    Skips if enough secondary clips already exist.
    """
    import scipy.io.wavfile
    import scipy.signal
    from tqdm import tqdm

    if not SECONDARY_VOICES:
        log.info("  No secondary voices configured — skipping secondary negatives")
        return

    primary_latin = _is_latin(" ".join(TARGET_PHRASES))
    sec_phrases = [p for p in NEGATIVE_PHRASES if _is_latin(p) != primary_latin]
    if not sec_phrases:
        log.info("  No secondary-script phrases in negative_phrases — skipping")
        return

    # Find secondary voice model files
    sec_voice_files = [
        MODELS_DIR / f"{prefix}.onnx"
        for prefix in SECONDARY_VOICES
        if (MODELS_DIR / f"{prefix}.onnx").exists()
    ]
    if not sec_voice_files:
        log.warning("  No secondary voice models found — skipping secondary negatives")
        return

    neg_train_dir = OUTPUT_DIR / MODEL_NAME / "negative_train"
    neg_test_dir  = OUTPUT_DIR / MODEL_NAME / "negative_test"
    neg_train_dir.mkdir(parents=True, exist_ok=True)
    neg_test_dir.mkdir(parents=True, exist_ok=True)

    existing_train = len(list(neg_train_dir.glob("sec_*.wav")))
    existing_test  = len(list(neg_test_dir.glob("sec_*.wav")))
    target_train, target_test = 1600, 400

    if existing_train >= target_train and existing_test >= target_test:
        log.info("  Secondary negatives already generated (%d train, %d test) — skipping",
                 existing_train, existing_test)
        return

    log.info("  Generating secondary negative clips: %d phrases × %d voices",
             len(sec_phrases), len(sec_voice_files))

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(PIPER_BIN_DIR.resolve()) + ":" + env.get("LD_LIBRARY_PATH", "")
    env["ESPEAK_DATA_PATH"] = str(PIPER_ESPEAK.resolve())

    length_scales = [0.8, 0.95, 1.0, 1.1, 1.25]
    noise_scales  = [0.667, 0.8, 0.98]

    import random
    combos = [
        (phrase, str(voice.resolve()), ls, ns)
        for phrase in sec_phrases
        for voice in sec_voice_files
        for ls in length_scales
        for ns in noise_scales
    ]
    random.shuffle(combos)

    total = target_train + target_test
    generated_train, generated_test = existing_train, existing_test
    pbar = tqdm(total=total, initial=existing_train + existing_test,
                desc="Secondary negatives", unit="clip")

    idx = 0
    while (generated_train < target_train or generated_test < target_test) and idx < len(combos) * 3:
        phrase, voice, ls, ns = combos[idx % len(combos)]
        idx += 1

        if generated_train < target_train:
            out_dir = neg_train_dir
            fname = f"sec_{generated_train:05d}.wav"
        elif generated_test < target_test:
            out_dir = neg_test_dir
            fname = f"sec_{generated_test:05d}.wav"
        else:
            break

        out_path = out_dir / fname
        result = subprocess.run(
            [str(PIPER_BIN.resolve()), "--model", voice,
             "--output_file", str(out_path),
             "--length_scale", str(ls),
             "--noise_scale", str(ns),
             "--noise_w", "0.8"],
            input=phrase, text=True, capture_output=True, env=env,
        )

        if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size <= 44:
            continue

        # Resample to 16 kHz if needed
        sr, data = scipy.io.wavfile.read(out_path)
        if sr != 16000:
            n_samples = int(len(data) * 16000 / sr)
            data = scipy.signal.resample(data, n_samples).astype(data.dtype)
            scipy.io.wavfile.write(str(out_path), 16000, data)

        if out_dir == neg_train_dir:
            generated_train += 1
        else:
            generated_test += 1
        pbar.update(1)

    pbar.close()
    log.info("  Secondary negatives: %d train, %d test", generated_train, generated_test)


def phase_generate():
    log.info("=" * 60)
    log.info("Phase: generate — generating TTS clips")
    log.info("=" * 60)

    # Generate secondary-language negative clips before the main generate_clips call
    _generate_secondary_negatives()

    # Log progress estimate
    pos_train_dir = OUTPUT_DIR / MODEL_NAME / "positive_train"
    existing = len(list(pos_train_dir.glob("*.wav"))) if pos_train_dir.exists() else 0
    delta = max(0, N_SAMPLES - existing)
    if delta > 0:
        log.info(
            "  %d clips exist, generating %d more (target: %d). "
            "Piper binary, CPU-only — ~0.2s/clip.",
            existing, delta, N_SAMPLES,
        )
    else:
        log.info("  %d clips already exist (target: %d) — generation will be skipped.",
                 existing, N_SAMPLES)

    yaml_path = write_yaml()
    run(f'"{PYTHON}" {TRAIN_SCRIPT} --training_config {yaml_path} --generate_clips')
    log.info("  Clip generation complete.")


# ===========================================================================
# PHASE 6: Augment clips (via openWakeWord train.py --augment_clips)
# ===========================================================================

def _ensure_oww_feature_models():
    """Download melspectrogram.onnx + embedding_model.onnx if missing.

    openWakeWord needs these models for feature extraction and augmentation.
    They are downloaded from the openWakeWord GitHub releases page.
    """
    models_dir = OWW_DIR / "openwakeword" / "resources" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    needed = ["melspectrogram.onnx", "embedding_model.onnx",
              "melspectrogram.tflite", "embedding_model.tflite"]
    missing = [f for f in needed if not (models_dir / f).exists()]
    if not missing:
        return
    log.info("  Downloading openWakeWord feature models (%s)...", ", ".join(missing))
    from openwakeword.utils import download_file
    base = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
    for f in missing:
        download_file(f"{base}/{f}", str(models_dir))
    log.info("  ✓ Feature models ready")


def _patch_augment_tqdm():
    """Patch openwakeword/data.py to add a tqdm progress bar to the augment loop.

    Without this, the augmentation phase appears to hang with no output.
    """
    data_py = OWW_DIR / "openwakeword" / "data.py"
    if not data_py.exists():
        return
    src = data_py.read_text()
    if "tqdm" in src:
        return  # already patched
    log.info("  Patching data.py: adding tqdm to augment_clips batch loop...")
    src = "from tqdm import tqdm\n" + src
    src = src.replace(
        "for i in range(0, len(clip_paths), batch_size)",
        'for i in tqdm(range(0, len(clip_paths), batch_size), desc="Augmenting", unit="batch")',
    )
    data_py.write_text(src)


def phase_augment():
    log.info("=" * 60)
    log.info("Phase: augment — augmenting clips with RIR + background noise")
    log.info("=" * 60)

    _patch_augment_tqdm()
    _ensure_oww_feature_models()

    # Check for partial or stale .npy feature files.
    # train.py only looks at positive_features_train.npy to decide whether to
    # skip augmentation — if that file exists but others are missing, augmentation
    # is silently skipped and training fails.  Similarly, if new clips were added
    # (e.g. punctuation variants), the existing .npy files are stale.
    feature_dir = OUTPUT_DIR / MODEL_NAME
    expected = [
        "positive_features_train.npy", "positive_features_test.npy",
        "negative_features_train.npy", "negative_features_test.npy",
    ]
    existing_npy = [f for f in expected if (feature_dir / f).exists()]

    stale = False
    if len(existing_npy) == 4:
        pos_dir = feature_dir / "positive_train"
        if pos_dir.exists():
            n_clips = len(list(pos_dir.glob("*.wav")))
            n_feat  = np.load(str(feature_dir / "positive_features_train.npy"),
                              mmap_mode="r").shape[0]
            if n_clips != n_feat:
                log.warning("Clip count changed (%d clips vs %d features) — "
                            "forcing re-augmentation", n_clips, n_feat)
                stale = True

    if existing_npy and (len(existing_npy) < 4 or stale):
        reason = "partial" if len(existing_npy) < 4 else "stale"
        log.warning("Deleting %s feature files (%d/4) to force clean augmentation",
                     reason, len(existing_npy))
        for f in existing_npy:
            (feature_dir / f).unlink()
            log.info("  Deleted %s", f)

    yaml_path = write_yaml()
    run(f'"{PYTHON}" {TRAIN_SCRIPT} --training_config {yaml_path} --augment_clips')

    # Verify all 4 feature files were actually created
    missing = [f for f in expected if not (feature_dir / f).exists()]
    if missing:
        raise RuntimeError(
            f"Augmentation subprocess exited 0 but {len(missing)} feature "
            f"file(s) are missing: {missing}"
        )
    log.info("  Augmentation complete — all 4 feature files created.")


def _patch_train_logging():
    """Ensure train.py calls logging.basicConfig(level=INFO).

    Without this, train.py's logging.info() calls (including final accuracy
    and recall metrics) are silently dropped because Python's default level
    is WARNING.
    """
    train_src = TRAIN_SCRIPT.read_text()
    marker = "logging.basicConfig(level=logging.INFO)"
    if marker not in train_src:
        log.info("  Patching train.py to enable INFO-level logging...")
        train_src = train_src.replace(
            "import logging\n",
            f"import logging\n{marker}\n",
            1,
        )
        TRAIN_SCRIPT.write_text(train_src)


# ===========================================================================
# PHASE 7: Train DNN model (via openWakeWord train.py --train_model)
# ===========================================================================

def phase_train():
    log.info("=" * 60)
    log.info("Phase: train — training DNN model (%d steps)", TRAINING_STEPS)
    log.info("=" * 60)

    yaml_path = write_yaml()
    # We do NOT pass --convert_to_tflite here; phase_export() handles
    # ONNX → TFLite conversion using onnx2tf.
    import re as _re

    # train.py never calls logging.basicConfig(), so metrics are lost at WARNING level
    _patch_train_logging()

    cmd = f'"{PYTHON}" {TRAIN_SCRIPT} --training_config {yaml_path} --train_model'
    log.info("$ %s", cmd)
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    output_lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        output_lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    # Extract and display final metrics prominently
    full_output = "".join(output_lines)
    accuracy = _re.search(r"Final Model Accuracy:\s*([\d.eE+-]+)", full_output)
    recall   = _re.search(r"Final Model Recall:\s*([\d.eE+-]+)", full_output)
    fp_hr    = _re.search(r"Final Model False Positives per Hour:\s*([\d.eE+-]+)", full_output)

    log.info("=" * 60)
    log.info("  TRAINING COMPLETE")
    log.info("=" * 60)
    if accuracy and recall and fp_hr:
        log.info("  Final Accuracy            : %s", accuracy.group(1))
        log.info("  Final Recall              : %s", recall.group(1))
        log.info("  Final False Positives/Hour: %s", fp_hr.group(1))
    else:
        log.warning("  Could not parse final metrics from training output")
    log.info("=" * 60)


# ===========================================================================
# PHASE 8: Export ONNX → TFLite
# ===========================================================================

def phase_export():
    """Convert ONNX model → TFLite using onnx2tf, copy to project root.

    Uses onnx2tf with keep_shape_absolutely_input_names to prevent the
    NCW → NWC dimension transposition that would break openWakeWord inference.
    """
    log.info("=" * 60)
    log.info("Phase: export — converting ONNX → TFLite")
    log.info("=" * 60)

    onnx_path   = OUTPUT_DIR / f"{MODEL_NAME}.onnx"
    tflite_dest = Path(".") / f"{MODEL_NAME}.tflite"

    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path}\n"
            "Run the train phase first."
        )

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        try:
            import onnx2tf
            log.info("  Converting ONNX → TFLite (onnx2tf)...")
            # Read the ONNX input tensor name to preserve its exact shape.
            # Without keep_shape_absolutely_input_names, onnx2tf transposes
            # NCW (1,16,96) → NWC (1,96,16), breaking openWakeWord inference.
            import onnx as _onnx
            _m = _onnx.load(str(onnx_path))
            _input_name = _m.graph.input[0].name
            onnx2tf.convert(
                input_onnx_file_path=str(onnx_path),
                output_folder_path=tmp,
                non_verbose=True,
                keep_shape_absolutely_input_names=[_input_name],
            )
            # onnx2tf produces <model_name>_float32.tflite
            tflite_src = Path(tmp) / f"{MODEL_NAME}_float32.tflite"
            if not tflite_src.exists():
                # fallback: find any .tflite
                candidates = list(Path(tmp).glob("*.tflite"))
                if candidates:
                    tflite_src = candidates[0]
                else:
                    raise FileNotFoundError("onnx2tf produced no .tflite file")

            shutil.copy(str(tflite_src), str(tflite_dest))
            size_kb = tflite_dest.stat().st_size / 1024
            log.info("  ✓ %s saved  (%.1f KB)  →  %s",
                     tflite_dest.name, size_kb, tflite_dest.resolve())
        except Exception as exc:
            log.warning("  TFLite conversion failed: %s", exc)
            log.warning(
                "The ONNX model is still usable with openWakeWord directly:\n"
                "  %s\n"
                "See: https://github.com/dscripka/openWakeWord#usage",
                onnx_path,
            )
            onnx_dest = Path(".") / f"{MODEL_NAME}.onnx"
            shutil.copy(str(onnx_path), str(onnx_dest))
            log.info("  Copied ONNX model as %s (tflite conversion skipped)", onnx_dest.name)


# ===========================================================================
# PREVIEW: Generate sample clips to verify pronunciation before full training
# ===========================================================================

def _run_preview(n: int):
    """Generate n sample clips into ./preview/ so the user can listen.

    Use this before starting a full training run to verify that the Piper
    voices pronounce the target phrase correctly.
    """
    import subprocess as _sp

    preview_dir = Path("preview")
    preview_dir.mkdir(exist_ok=True)

    if not MODELS_DIR.exists() or not list(MODELS_DIR.glob("*.onnx")):
        log.info("Voices not yet downloaded — fetching first...")
        phase_voices()

    if not PIPER_BIN.exists():
        raise FileNotFoundError(
            f"Piper binary not found: {PIPER_BIN}\nRun ./setup.sh first."
        )

    voice_files = sorted(MODELS_DIR.glob("*.onnx"))
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(PIPER_BIN_DIR.resolve()) + ":" + env.get("LD_LIBRARY_PATH", "")
    env["ESPEAK_DATA_PATH"] = str(PIPER_ESPEAK.resolve())

    length_scales = [0.8, 0.95, 1.0, 1.1, 1.25]

    log.info("Generating %d preview clips of %s → ./preview/", n, TARGET_PHRASES)
    for i in range(n):
        voice = voice_files[i % len(voice_files)]
        ls = length_scales[i % len(length_scales)]
        phrase = TARGET_PHRASES[i % len(TARGET_PHRASES)]
        out = preview_dir / f"{i+1:02d}_{voice.stem}_speed{ls}.wav"

        result = _sp.run(
            [str(PIPER_BIN.resolve()), "--model", str(voice.resolve()),
             "--output_file", str(out),
             "--length_scale", str(ls),
             "--noise_scale", "0.667",
             "--noise_w", "0.8"],
            input=phrase, text=True, capture_output=True, env=env,
        )
        if result.returncode == 0 and out.exists() and out.stat().st_size > 44:
            log.info("  ✓ %s  (%.1f KB)", out.name, out.stat().st_size / 1024)
        else:
            log.warning("  ✗ %s  failed: %s", out.name, result.stderr.strip())

    log.info("")
    log.info("Listen to the clips in ./preview/ and verify the pronunciation.")
    log.info("If they sound right, run the full training:")
    log.info("  ./run_training.sh --config <config.yaml>")


# ===========================================================================
# MAIN
# ===========================================================================

ALL_PHASES = ["setup", "voices", "features", "background", "speech",
              "generate", "augment", "train", "export"]


def main():
    parser = argparse.ArgumentParser(
        description="Train a custom wake word model using openWakeWord",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Phases (run in order for a fresh training):
              setup        Write generate_samples wrapper (needed by train.py)
              voices       Download Piper ONNX voice models
              features     Download ACAV100M + validation .npy files (~7 GB)
              background   Download MIT RIRs, AudioSet, FMA background audio
              speech       Extract real speech features as a negative class
              generate     TTS clip generation via openWakeWord train.py
              augment      Noise + RIR augmentation via openWakeWord train.py
              train        DNN training (outputs training/output/<name>.onnx)
              export       ONNX → TFLite → <model_name>.tflite at project root

            Quick start:
              ./setup.sh
              ./run_training.sh --config example_ru_jarvis.yaml

            Skip ACAV100M download if files already present:
              ./run_training.sh --config example_ru_jarvis.yaml --skip-features

            Preview TTS pronunciation before committing to full training:
              ./run_training.sh --config example_ru_jarvis.yaml --preview 5
        """),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (e.g. example_ru_jarvis.yaml)",
    )
    parser.add_argument(
        "--phase",
        choices=["all"] + ALL_PHASES + ["ru_speech"],  # ru_speech = legacy alias
        default="all",
        help="Which phase to run (default: all)",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip ACAV100M download (use if files already present in training/features/)",
    )
    parser.add_argument(
        "--preview", type=int, nargs="?", const=5, metavar="N",
        help="Generate N preview clips (default 5) into ./preview/ and exit. "
             "Use this to verify TTS pronunciation before starting full training.",
    )
    args = parser.parse_args()
    if args.phase == "ru_speech":
        args.phase = "speech"

    cfg = load_config(args.config)
    _apply_config(cfg)

    if args.preview is not None:
        _run_preview(args.preview)
        return

    run_all = args.phase == "all"

    log.info("=" * 60)
    log.info("openWakeWord Trainer — %s", MODEL_NAME)
    log.info("  Target phrases : %s", TARGET_PHRASES)
    log.info("  Samples        : %d train + %d val", N_SAMPLES, N_SAMPLES_VAL)
    log.info("  Training steps : %d", TRAINING_STEPS)
    log.info("  Neg. penalty   : %d", MAX_NEG_WEIGHT)
    log.info("=" * 60)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if run_all or args.phase == "setup":
        phase_setup()

    if run_all or args.phase == "voices":
        phase_voices()

    if args.phase == "features":
        if not args.skip_features:
            phase_features()
    elif run_all and not args.skip_features:
        phase_features()
    elif run_all and args.skip_features:
        log.info("Skipping feature download (--skip-features)")

    if run_all or args.phase == "background":
        phase_background()

    if args.phase == "speech":
        phase_ru_speech()
    elif run_all:
        phase_ru_speech()

    if run_all or args.phase == "generate":
        phase_generate()

    if run_all or args.phase == "augment":
        phase_augment()

    if run_all or args.phase == "train":
        phase_train()

    if run_all or args.phase == "export":
        phase_export()

    if run_all:
        log.info("=" * 60)
        log.info("All phases complete!")
        tflite = Path(f"{MODEL_NAME}.tflite")
        onnx_  = Path(f"{MODEL_NAME}.onnx")
        if tflite.exists():
            log.info("  ✓  %s  (%.1f KB)", tflite, tflite.stat().st_size / 1024)
        elif onnx_.exists():
            log.info("  ✓  %s  (%.1f KB) — TFLite conversion failed; ONNX is usable",
                     onnx_, onnx_.stat().st_size / 1024)
        log.info("=" * 60)


if __name__ == "__main__":
    main()

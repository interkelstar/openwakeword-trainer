# openWakeWord Trainer

Config-driven pipeline for training custom wake word models using the
[openWakeWord](https://github.com/dscripka/openWakeWord) framework.

openWakeWord trains a small DNN on top of pre-computed ACAV100M audio
embeddings. The result is an ~860 KB TFLite model that runs in real time on
CPU and integrates with [linux-voice-assistant](https://github.com/your-lva-repo)
and other openWakeWord-compatible runtimes.

## Requirements

- Linux (tested on Ubuntu 22.04, Debian 12, WSL2)
- Python 3.9+
- ~26 GB disk space for training data
- NVIDIA GPU optional but cuts training time from ~10 hours to ~2 hours

## Quick Start

```bash
# Clone the repo and set up the environment (one time only)
git clone https://github.com/interkelstar/openwakeword-trainer-ru.git
cd openwakeword-trainer-ru
chmod +x setup.sh && ./setup.sh

# Copy the example config and customise it for your wake word
cp example_ru_jarvis.yaml my_wakeword.yaml
# Edit my_wakeword.yaml: set model_name, target_phrases, voices, negatives

# (Optional) Listen to a few TTS samples before committing to a full run
./run_training.sh --config my_wakeword.yaml --preview 5

# Run the full training pipeline (~10–17 hours on CPU, ~2–4 hours with GPU)
./run_training.sh --config my_wakeword.yaml
```

Output files created at the project root:

- `<model_name>.tflite` — TFLite model ready for use with openWakeWord
- `<model_name>.onnx` — ONNX model (fallback if TFLite conversion fails)

## Training Phases

The pipeline runs these phases in order:

| Phase        | What it does                                                   | Time (CPU)  |
|--------------|----------------------------------------------------------------|-------------|
| `setup`      | Write generate_samples wrapper for openWakeWord train.py       | seconds     |
| `voices`     | Download Piper ONNX voice models from HuggingFace              | 1–5 min     |
| `features`   | Download ACAV100M + validation feature files (~7 GB)           | 30–60 min   |
| `background` | Download MIT RIRs + AudioSet + FMA background audio (~5 GB)    | 30–60 min   |
| `ru_speech`  | Extract Russian speech features as a dedicated negative class  | 10–30 min   |
| `generate`   | Generate TTS clips via piper binary                            | 6–10 hours  |
| `augment`    | Augment clips with room impulse responses + background noise   | 2–4 hours   |
| `train`      | Train the DNN model                                            | 1–12 hours  |
| `export`     | Convert ONNX → TFLite, copy to project root                    | 5–10 min    |

Run individual phases with `--phase <name>`:

```bash
./run_training.sh --config my_wakeword.yaml --phase train
./run_training.sh --config my_wakeword.yaml --phase export
```

Skip the 7 GB ACAV100M download if you already have the files:

```bash
./run_training.sh --config my_wakeword.yaml --skip-features
```

## Config Reference

See `example_ru_jarvis.yaml` for a fully annotated example. Key fields:

```yaml
model_name: ru_jarvis          # Output filename prefix (ASCII, no spaces)
target_phrases:                # What the model should fire on
  - Джарвис
  - Джарвис!
  - Джарвис?

training:
  n_samples: 50000             # TTS clips for training (25k min, 100k+ for production)
  n_samples_val: 10000         # TTS clips for validation
  steps: 500000                # Gradient steps (increase if recall is too low)
  layer_size: 128              # DNN hidden layer width (64–192)
  max_neg_weight: 3000         # False positive penalty (raise to reduce FP rate)

voices:
  primary:                     # For positive clips + same-language negatives
    base_url: https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU
    models:
      ru_RU-dmitri-medium: dmitri/medium
  secondary:                   # For cross-language negative phrases (optional)
    base_url: https://huggingface.co/rhasspy/piper-voices/resolve/main
    models:
      en_US-amy-medium: en/en_US/amy/medium

negative_phrases:              # Acoustically similar words the model should reject
  - Джар
  - Джек
  - Алиса
  - Jarring                    # Latin script → synthesised with secondary voices

russian_speech_negatives:
  enabled: true                # Include real Russian speech as a negative class
  dataset: bond005/sberdevices_golos_10h_crowd
  max_chunks: 5000
```

### How negative phrases work

The pipeline synthesises `negative_phrases` as TTS clips and trains the model
to reject them. This dramatically reduces false activations in real use.

- **Primary language phrases** (Cyrillic in the example): synthesised with
  the primary voices, same as positive clips.
- **Secondary language phrases** (Latin script): synthesised with secondary
  voices. Useful for reducing false positives from TV/YouTube in a different
  language.
- **Russian speech negatives**: real speech from a HuggingFace dataset,
  extracted as pre-computed embeddings. Teaches the model to reject general
  conversation, not just the specific adversarial phrases.

### Choosing voices

Browse available voices at https://huggingface.co/rhasspy/piper-voices

More voices = more speaker diversity = better generalisation. Use "medium"
quality (the "low" models are faster to synthesise but less natural).

The voice path format is `<lang>/<lang_region>/<name>/<quality>/<name>`:

```yaml
voices:
  primary:
    base_url: https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE
    models:
      de_DE-thorsten-medium: thorsten/medium
```

## Output Files

After the `export` phase completes:

- `<model_name>.tflite` — the trained model (~860 KB)
- `<model_name>.onnx` — the ONNX model (if TFLite conversion failed)

The ONNX model is usable directly with openWakeWord without the TFLite step.
See [openWakeWord usage docs](https://github.com/dscripka/openWakeWord#usage).

## ElevenLabs High-Quality Positives (Optional)

`generate_elevenlabs.py` generates additional high-quality TTS clips using the
ElevenLabs API (requires an API key with credits). These are mixed into the
positive training set during feature extraction and improve naturalness:

```bash
.venv/bin/python generate_elevenlabs.py --api-key YOUR_KEY --config my_wakeword.yaml
```

Clips are saved to `training/output/<model_name>/elevenlabs_positive/` and
will be automatically picked up the next time you run `--phase augment`.

## Disk Space

After a full training run, the `training/` directory contains:

```
training/repos/          # piper-sample-generator + openWakeWord (~500 MB)
training/piper_binary/   # piper executable + shared libs (~150 MB)
training/piper_models/   # downloaded ONNX voice models (~280 MB for 4 voices)
training/features/       # ACAV100M + validation .npy files (~7 GB)
training/mit_rirs/       # room impulse responses (~15 MB)
training/background/     # AudioSet + FMA (~5 GB)
training/output/         # TTS clips + augmented features (~10 GB)
```

To clean up everything after training:

```bash
rm -rf .venv training/
```

## How It Works

openWakeWord's approach:

1. A frozen embedding model (not trained here) converts raw audio into
   96-dimensional embeddings at 12.5 Hz.
2. A small DNN (trained here) classifies sequences of 16 frames (1.28 s) as
   wake-word / not-wake-word.
3. The DNN is trained against:
   - **Positive**: TTS clips of the target phrase
   - **ACAV100M negatives**: 2000 hours of pre-computed audio embeddings
   - **Adversarial negatives**: TTS clips of acoustically similar words
   - **Russian speech negatives** (optional): real speech embeddings

## License

MIT

from pathlib import Path

from src.utils.download import resolve_input
from src.utils.io_utils import DEMO_DATA_DIR, ROOT_PATH


def prepare_demo(model_source=None, audio_source=None):
    DEMO_DATA_DIR.mkdir(exist_ok=True)

    print("Resolving files... It might take a few minutes.")
    if model_source is not None:
        model_path = resolve_input(model_source, DEMO_DATA_DIR, filename="model.pth")
    else:
        model_path = None

    if audio_source is not None:
        audio_path = resolve_input(audio_source, DEMO_DATA_DIR)
    else:
        audio_path = None

    print("Done.")
    return model_path, audio_path


MODEL_SOURCE = "https://huggingface.co/buckets/ulorew/NeuralCodec/resolve/model.pth?download=true"
AUDIO_SOURCE = "https://huggingface.co/buckets/ulorew/NeuralCodec/resolve/helloeveryone%E2%80%8B.flac?download=true"

if __name__ == "__main__":
    prepare_demo(model_source=MODEL_SOURCE, audio_source=AUDIO_SOURCE)

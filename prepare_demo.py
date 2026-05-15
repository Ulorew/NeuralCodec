from pathlib import Path
from src.utils.download import resolve_input
from src.utils.io_utils import ROOT_PATH, DEMO_DATA_DIR


def prepare_demo(model_source, audio_source):
    DEMO_DATA_DIR.mkdir(exist_ok=True)

    print("Resolving files... It might take a few minutes.")
    model_path = resolve_input(model_source, DEMO_DATA_DIR, filename="model_best.pth")
    audio_path = resolve_input(audio_source, DEMO_DATA_DIR)
    print("Done.")
    return model_path, audio_path


MODEL_SOURCE = "https://huggingface.co/buckets/ulorew/NeuralCodec/resolve/gan_training_bit6_200ep_0.5s_mixed_loss_C32_great_best.pth?download=true"
AUDIO_SOURCE = "https://huggingface.co/buckets/ulorew/NeuralCodec/resolve/helloeveryone%E2%80%8B.flac?download=true"

if __name__ == "__main__":
    prepare_demo(model_source=MODEL_SOURCE, audio_source=AUDIO_SOURCE)


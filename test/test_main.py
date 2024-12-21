import pytest
import numpy as np
from main.main import load_model, generate_speech


@pytest.fixture(scope="module")
def model_setup():
    model, processor, device = load_model()
    return model, processor, device


def test_load_model(model_setup):
    model, processor, device = model_setup
    assert model is not None
    assert processor is not None
    assert device in ["cuda:0", "cpu"]


def test_generate_speech(model_setup):
    model, processor, device = model_setup
    text_prompt = "Hello, this is a test."

    audio_array = generate_speech(model, processor, device, text_prompt)
    assert isinstance(audio_array, np.ndarray), "Audio output should be a numpy array."
    assert audio_array.size > 0, "Audio output should not be empty."

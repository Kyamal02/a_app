import pytest
import numpy as np
import torch
from app.main.main import load_model, generate_speech


def test_load_model():
    model, processor, device = load_model()
    assert model is not None
    assert processor is not None
    # Проверка что device - либо cuda, либо cpu
    assert device in ["cuda:0", "cpu"]


def test_generate_speech():
    model, processor, device = load_model()
    text_prompt = "Hello, this is a test."

    audio_array = generate_speech(model, processor, device, text_prompt)
    assert isinstance(audio_array, np.ndarray), "Audio output should be a numpy array."
    assert audio_array.size > 0, "Audio output should not be empty."

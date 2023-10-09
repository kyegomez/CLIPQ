import torch
import pytest
from PIL import Image
from clipq.main import CLIPQ

@pytest.fixture
def clipq():
    return CLIPQ()

@pytest.fixture
def test_image():
    return Image.open("agorabanner.png")

def test_fetch_image_from_url():
    clipq = CLIPQ()
    image = clipq.fetch_image_from_url()
    assert isinstance(image, Image.Image)

def test_load_image_from_path():
    clipq = CLIPQ()
    image = clipq.load_image_from_path("agorabanner.png")
    assert isinstance(image, Image.Image)

def test_split_image():
    clipq = CLIPQ()
    image = Image.new("RGB", (800, 600))
    slices = clipq.split_image(image, h_splits=2, v_splits=2)
    assert len(slices) == 4

def test_get_vectors(test_image):
    clipq = CLIPQ()
    vectors = clipq.get_vectors(test_image, h_splits=2, v_splits=2)
    assert len(vectors) == 4

def test_run_from_url():
    clipq = CLIPQ()
    vectors = clipq.run_from_url(h_splits=2, v_splits=2)
    assert len(vectors) == 4

def test_run_from_path(test_image):
    clipq = CLIPQ()
    vectors = clipq.run_from_path(path="agorabanner.png", h_splits=2, v_splits=2)
    assert len(vectors) == 4

def test_check_hard_chunking():
    clipq = CLIPQ()
    quadrants = [
        torch.zeros((10, 10)),
        torch.ones((10, 10)),
        torch.ones((10, 10)),
        torch.zeros((10, 10)),
    ]
    variances = clipq.check_hard_chunking(quadrants)
    assert variances == [0.0, 0.0, 0.0, 0.0]

def test_embed_whole_image(test_image):
    clipq = CLIPQ()
    image_embed = clipq.embed_whole_image(test_image)
    assert image_embed.shape == (512,)

def test_apply_noise_reduction(test_image):
    clipq = CLIPQ()
    blurred_image = clipq.apply_noise_reduction(test_image, kernel_size=5)
    assert isinstance(blurred_image, Image.Image)

def test_get_captions(test_image):
    clipq = CLIPQ()
    candidate_captions = ["a serene landscape", "a group of people", "an animal in the wild", "a building", "vehicles on the road"]
    caption = clipq.get_captions(test_image, candidate_captions)
    assert isinstance(caption, str)

def test_get_and_concat_captions(test_image):
    clipq = CLIPQ()
    candidate_captions = ["a serene landscape", "a group of people", "an animal in the wild", "a building", "vehicles on the road"]
    concated_captions = clipq.get_and_concat_captions(test_image, candidate_captions, h_splits=2, v_splits=2)
    assert isinstance(concated_captions, str)
from io import BytesIO

import requests
import torch
from PIL import Image
from torchvision.transforms import GaussianBlur
from transformers import CLIPModel, CLIPProcessor


# clipq/main.py
class CLIPQ:
    """
    ClipQ

    A class to run CLIP model on images and get the vectors.

    ...

    Attributes
    ----------
    model_name : str
        The name of the CLIP model to use.
    query_text : str
        The query text to use.
    model : transformers.CLIPModel
        The CLIP model.
    processor : transformers.CLIPProcessor
        The CLIP processor.

    Methods
    -------
    fetch_image_from_url(url)
        Fetches an image from the given url.

    load_image_from_path(path)
        Loads an image from the given path.

    split_image(image, h_splits, v_splits)
        Splits the given image into h_splits x v_splits parts.

    get_vectors(image, h_splits, v_splits)
        Gets the vectors for the given image.

    run_from_url(url, h_splits, v_splits)
        Runs the model on the image fetched from the given url.

    run_from_path(path, h_splits, v_splits)
        Runs the model on the image loaded from the given path.

    Usage for simple split
    test = CLIPQ(query_text="A photo of a cat")
    vectors = test.run_from_url(h_splits=3, v_splits=3)
    print(vectors)



    Usage for captions
    test = CLIPQ()

    candidates = ["a serene landscape", "a group of people", "an animal in the wild", "a building", "vehicles on the road"]


    caption_result = experiment.get_and_concat_captions(experiment.fetch_image_from_url(), candidates)
    print(caption_result)


    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        query_text: str = "A photo ",
    ):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.query_text = query_text

    def fetch_image_from_url(self, url: str = "https://picsum.photos/800"):
        """Fetches an image from the given url"""
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception("Failed to fetch an image")
        image = Image.open(BytesIO(response.content))
        return image

    def load_image_from_path(self, path):
        """Loads an image from the given path"""
        return Image.open(path)

    def split_image(self, image, h_splits: int = 2, v_splits: int = 2):
        """Splits the given image into h_splits x v_splits parts"""
        width, height = image.size
        w_step, h_step = width // h_splits, height // v_splits
        slices = []

        for i in range(v_splits):
            for j in range(h_splits):
                slice = image.crop(
                    (j * w_step, i * h_step, (j + 1) * w_step, (i + 1) * h_step)
                )
                slices.append(slice)
        return slices

    def get_vectors(
        self,
        image,
        h_splits: int = 2,
        v_splits: int = 2,
    ):
        """Gets the vectors for the given image"""
        slices = self.split_image(image, h_splits, v_splits)
        vectors = []

        for slice in slices:
            inputs = self.processor(
                text=self.query_text, images=slice, return_tensors="pt", padding=True
            )
            outputs = self.model(**inputs)
            vectors.append(outputs.image_embeds.squeeze().detach().numpy())
        return vectors

    def run_from_url(
        self,
        url: str = "https://picsum.photos/800",
        h_splits: int = 2,
        v_splits: int = 2,
    ):
        """Runs the model on the image fetched from the given url"""
        image = self.fetch_image_from_url(url)
        return self.get_vectors(image, h_splits, v_splits)

    def check_hard_chunking(self, quadrants):
        """Check if the chunking is hard"""
        variances = []
        for quadrant in quadrants:
            edge_pixels = torch.cat(
                [
                    quadrant[0, 1],
                    quadrant[-1, :],
                ]
            )
            variances.append(torch.var(edge_pixels).item())
        return variances

    def embed_whole_image(self, image):
        """Embed the entire image"""
        inputs = self.model(
            image,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            return outputs.image_embeds.squeeze()

    def apply_noise_reduction(self, image, kernel_size: int = 5):
        """Implement an upscaling method to upscale the image and tiling issues"""
        blur = GaussianBlur(kernel_size)
        return blur(image)

    def run_from_path(self, path: str = None, h_splits: int = 2, v_splits: int = 2):
        """Runs the model on the image loaded from the given path"""
        image = self.load_image_from_path(path)
        return self.get_vectors(image, h_splits, v_splits)

    def get_captions(self, image, candidate_captions):
        """Get the best caption for the given image"""
        inputs_image = self.processor(
            text=[self.query_text], images=image, return_tensors="pt", padding=True
        )

        inputs_text = self.processor(
            text=candidate_captions, return_tensors="pt", padding=True
        )

        image_embeds = self.model(**inputs_image).image_embeds
        text_embeds = self.model(**inputs_text).text_embeds

        # cal similarity between image and text
        similarities = (image_embeds @ text_embeds.T).squeeze(0)
        best_caption_index = similarities.argmax().item()

        return candidate_captions[best_caption_index]

    def get_and_concat_captions(
        self, image, candidate_captions, h_splits=2, v_splits=2
    ):
        """Get the best caption for the given image"""
        slices = self.split_image(image, h_splits, v_splits)
        captions = [self.get_captions(slice, candidate_captions) for slice in slices]
        concated_captions = "".join(captions)
        return concated_captions

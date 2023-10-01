import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO

class CLIPQ:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    def fetch_random_image(self):
        # For simplicity, we'll fetch a random image from Lorem Picsum
        response = requests.get('https://picsum.photos/800')  # 800x800 image
        if response.status_code != 200:
            raise Exception("Failed to fetch an image.")
        image = Image.open(BytesIO(response.content))
        return image

    def split_image_into_quadrants(self, image):
        width, height = image.size
        quadrants = [
            image.crop((0, 0, width/2, height/2)),
            image.crop((width/2, 0, width, height/2)),
            image.crop((0, height/2, width/2, height)),
            image.crop((width/2, height/2, width, height))
        ]
        return quadrants

    def get_quadrant_vectors(self, image):
        quadrants = self.split_image_into_quadrants(image)
        
        quadrant_vectors = []
        for quadrant in quadrants:
            inputs = self.processor(
                text="a photo", 
                images=quadrant, 
                return_tensors="pt", 
                padding=True
            )

            outputs = self.model(**inputs)

            quadrant_vectors.append(outputs.image_embeds.squeeze().detach().numpy())
        return quadrant_vectors

    def run(self):
        image = self.fetch_random_image()
        vectors = self.get_quadrant_vectors(image)
        return vectors

# Using the class
experiment = CLIPQ()
vectors = experiment.run()
print(vectors)  # this will print the vectors for each quadrant
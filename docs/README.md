# CLIPQ Class


The `CLIPQ` class is designed to run the CLIP model on images and extract the vectors. It provides methods to fetch images from a URL or load them from a path, split the images into multiple parts, and get the vectors for each part. It also provides methods to run the model on the image fetched from a URL or loaded from a path.

## Attributes
-   `model_name` (str): The name of the CLIP model to use. Default is "openai/clip-vit-base-patch16".
-   `query_text` (str): The query text to use. Default is "A photo ".
-   `model` (transformers.CLIPModel): The CLIP model.
-   `processor` (transformers.CLIPProcessor): The CLIP processor.

## Methods

### `__init__(self, model_name: str = "openai/clip-vit-base-patch16", query_text: str = "A photo ")`

The constructor for the `CLIPQ` class. Initializes the CLIP model and processor.

#### Parameters

-   `model_name` (str, optional): The name of the CLIP model to use. Default is "openai/clip-vit-base-patch16".
-   `query_text` (str, optional): The query text to use. Default is "A photo ".

#### Example

```
clipq = CLIPQ(model_name="openai/clip-vit-base-patch16", query_text="A photo of a cat")
```



### `fetch_image_from_url(self, url='https://picsum.photos/800')`

Fetches an image from the given URL.

#### Parameters

-   `url` (str, optional): The URL to fetch the image from. Default is '<https://picsum.photos/800>'.

#### Returns

-   `image` (PIL.Image.Image): The fetched image.

#### Example

```
image = clipq.fetch_image_from_url(url="https://example.com/image.jpg")
```



### `load_image_from_path(self, path)`

Loads an image from the given path.

#### Parameters

-   `path` (str): The path to load the image from.

#### Returns

-   `image` (PIL.Image.Image): The loaded image.

#### Example

```
image = clipq.load_image_from_path(path="/path/to/image.jpg")
```



### `split_image(self, image, h_splits: int = 2, v_splits: int = 2)`

Splits the given image into `h_splits` x `v_splits` parts.

#### Parameters

-   `image` (PIL.Image.Image): The image to split.
-   `h_splits` (int, optional): The number of horizontal splits. Default is 2.
-   `v_splits` (int, optional): The number of vertical splits. Default is 2.

#### Returns

-   `slices` (list of PIL.Image.Image): The list of image slices.

#### Example

```
slices = clipq.split_image(image, h_splits=3, v_splits=3)
```



### `get_vectors(self, image, h_splits: int = 2, v_splits: int = 2)`

Gets the vectors for the given image.

#### Parameters

-   `image` (PIL.Image.Image): The image to get vectors for.
-   `h_splits` (int, optional): The number of horizontal splits. Default is 2.
-   `v_splits` (int, optional): The number of vertical splits. Default is 2.

#### Returns

-   `vectors` (list of numpy.ndarray): The list of vectors for each image slice.

#### Example

```
vectors = clipq.get_vectors(image, h_splits=3, v_splits=3)
```



### `run_from_url(self, url: str = 'https://picsum.photos/800', h_splits: int = 2, v_splits: int = 2)`

Runs the model on the image fetched from the given URL.

#### Parameters

-   `url` (str, optional): The URL to fetch the image from. Default is '<https://picsum.photos/800>'.
-   `h_splits` (int, optional): The number of horizontal splits. Default is 2.
-   `v_splits` (int, optional): The number of vertical splits. Default is 2.

#### Returns

-   `vectors` (list of numpy.ndarray): The list of vectors for each image slice.

#### Example

```
vectors = clipq.run_from_url(url="https://example.com/image.jpg", h_splits=3, v_splits=3)
```



### `run_from_path(self, path: str = None, h_splits: int = 2, v_splits: int =

You:

KEEP GOING DO NOT STOP

Athena:

### `run_from_path(self, path: str = None, h_splits: int = 2, v_splits: int = 2)`

Runs the model on the image loaded from the given path.

#### Parameters

-   `path` (str, optional): The path to load the image from.
-   `h_splits` (int, optional): The number of horizontal splits. Default is 2.
-   `v_splits` (int, optional): The number of vertical splits. Default is 2.

#### Returns

-   `vectors` (list of numpy.ndarray): The list of vectors for each image slice.

#### Example

```
vectors = clipq.run_from_path(path="/path/to/image.jpg", h_splits=3, v_splits=3)
```



### `get_captions(self, image, candidate_captions)`

Get the best caption for the given image.

#### Parameters

-   `image` (PIL.Image.Image): The image to get caption for.
-   `candidate_captions` (list of str): The list of candidate captions.

#### Returns

-   `best_caption` (str): The best caption for the image.

#### Example

```
candidates = ["a serene landscape", "a group of people", "an animal in the wild", "a building", "vehicles on the road"]
best_caption = clipq.get_captions(image, candidate_captions=candidates)
```



### `get_and_concat_captions(self, image, candidate_captions, h_splits=2, v_splits=2)`

Get the best caption for each slice of the given image and concatenate them.

#### Parameters

-   `image` (PIL.Image.Image): The image to get caption for.
-   `candidate_captions` (list of str): The list of candidate captions.
-   `h_splits` (int, optional): The number of horizontal splits. Default is 2.
-   `v_splits` (int, optional): The number of vertical splits. Default is 2.

#### Returns

-   `concated_captions` (str): The concatenated captions for the image slices.

#### Example

```
candidates = ["a serene landscape", "a group of people", "an animal in the wild", "a building", "vehicles on the road"]
concated_captions = clipq.get_and_concat_captions(image, candidate_captions=candidates, h_splits=3, v_splits=3)
```



### `check_hard_chunking(self, quadrants)`

Check if the chunking is hard.

#### Parameters

-   `quadrants` (list of numpy.ndarray): The list of quadrants.

#### Returns

-   `variances` (list of float): The list of variances for each quadrant.

#### Example

```
variances = clipq.check_hard_chunking(quadrants)
```



### `embed_whole_image(self, image)`

Embed the entire image.

#### Parameters

-   `image` (PIL.Image.Image): The image to embed.

#### Returns

-   `image_embeds` (torch.Tensor): The image embeddings.

#### Example

```
image_embeds = clipq.embed_whole_image(image)
```



### `apply_noise_reduction(self, image, kernel_size: int = 5)`

Implement an upscaling method to upscale the image and tiling issues.

#### Parameters

-   `image` (PIL.Image.Image): The image to apply noise reduction to.
-   `kernel_size` (int, optional): The kernel size for the Gaussian blur. Default is 5.

#### Returns

-   `blurred_image` (PIL.Image.Image): The blurred image.

#### Example

```
blurred_image = clipq.apply_noise_reduction(image, kernel_size=7)
```



You:

KEEP GOING DO NOT STOP

Athena:

Usage Examples
--------------

### Example 1: Simple Split

In this example, we will fetch an image from a URL, split it into 3x3 parts, and get the vectors for each part.

```
test = CLIPQ(query_text="A photo of a cat")
vectors = test.run_from_url(url="https://example.com/image.jpg", h_splits=3, v_splits=3)
print(vectors)
```



### Example 2: Captions

In this example, we will fetch an image from a URL, split it into 2x2 parts, get the best caption for each part, and concatenate them.

```
test = CLIPQ()
candidates = ["a serene landscape", "a group of people", "an animal in the wild", "a building", "vehicles on the road"]
image = test.fetch_image_from_url(url="https://example.com/image.jpg")
caption_result = test.get_and_concat_captions(image, candidate_captions=candidates, h_splits=2, v_splits=2)
print(caption_result)
```



### Example 3: Noise Reduction

In this example, we will fetch an image from a URL, apply noise reduction to it, and get the vectors for the entire image.

```
test = CLIPQ()
image = test.fetch_image_from_url(url="https://example.com/image.jpg")
blurred_image = test.apply_noise_reduction(image, kernel_size=7)
image_embeds = test.embed_whole_image(blurred_image)
print(image_embeds)
```



### Example 4: Hard Chunking

In this example, we will fetch an image from a URL, split it into 2x2 parts, and check if the chunking is hard.

```
test = CLIPQ()
image = test.fetch_image_from_url(url="https://example.com/image.jpg")
slices = test.split_image(image, h_splits=2, v_splits=2)
variances = test.check_hard_chunking(slices)
print(variances)
```



### Example 5: Run from Path

In this example, we will load an image from a path, split it into 3x3 parts, and get the vectors for each part.

```
test = CLIPQ(query_text="A photo of a cat")
vectors = test.run_from_path(path="/path/to/image.jpg", h_splits=3, v_splits=3)
print(vectors)
```



You:

Keep going do not stop

Athena:

### Example 6: Get Captions

In this example, we will fetch an image from a URL and get the best caption for it from a list of candidate captions.

```
test = CLIPQ()
candidates = ["a serene landscape", "a group of people", "an animal in the wild", "a building", "vehicles on the road"]
image = test.fetch_image_from_url(url="https://example.com/image.jpg")
best_caption = test.get_captions(image, candidate_captions=candidates)
print(best_caption)
```



### Example 7: Embed Whole Image

In this example, we will fetch an image from a URL and get the embeddings for the entire image.

```
test = CLIPQ()
image = test.fetch_image_from_url(url="https://example.com/image.jpg")
image_embeds = test.embed_whole_image(image)
print(image_embeds)
```



### Example 8: Load Image from Path

In this example, we will load an image from a path and get the embeddings for the entire image.

```
test = CLIPQ()
image = test.load_image_from_path(path="/path/to/image.jpg")
image_embeds = test.embed_whole_image(image)
print(image_embeds)
```



### Example 9: Split Image

In this example, we will fetch an image from a URL, split it into 4x4 parts, and get the vectors for each part.

```
test = CLIPQ(query_text="A photo of a cat")
image = test.fetch_image_from_url(url="https://example.com/image.jpg")
slices = test.split_image(image, h_splits=4, v_splits=4)
vectors = [test.get_vectors(slice) for slice in slices]
print(vectors)
```



### Example 10: Apply Noise Reduction

In this example, we will fetch an image from a URL, apply noise reduction to it with a kernel size of 9, and get the vectors for the entire image.

```
test = CLIPQ()
image = test.fetch_image_from_url(url="https://example.com/image.jpg")
blurred_image = test.apply_noise_reduction(image, kernel_size=9)
image_embeds = test.embed_whole_image(blurred_image)
print(image_embeds)
```



You:

Keep going do not stop

Athena:

### Example 11: Get and Concat Captions from Image Path

In this example, we will load an image from a path, split it into 2x2 parts, get the best caption for each part from a list of candidate captions, and concatenate them.

```
test = CLIPQ()
candidates = ["a serene landscape", "a group of people", "an animal in the wild", "a building", "vehicles on the road"]
image = test.load_image_from_path(path="/path/to/image.jpg")
caption_result = test.get_and_concat_captions(image, candidate_captions=candidates, h_splits=2, v_splits=2)
print(caption_result)
```



### Example 12: Check Hard Chunking from Image Path

In this example, we will load an image from a path, split it into 2x2 parts, and check if the chunking is hard.

```
test = CLIPQ()
image = test.load_image_from_path(path="/path/to/image.jpg")
slices = test.split_image(image, h_splits=2, v_splits=2)
variances = test.check_hard_chunking(slices)
print(variances)
```



### Example 13: Fetch Image from URL and Get Vectors

In this example, we will fetch an image from a URL and get the vectors for the entire image.

```
test = CLIPQ()
image = test.fetch_image_from_url(url="https://example.com/image.jpg")
vectors = test.get_vectors(image)
print(vectors)
```



### Example 14: Load Image from Path and Get Vectors

In this example, we will load an image from a path and get the vectors for the entire image.

```
test = CLIPQ()
image = test.load_image_from_path(path="/path/to/image.jpg")
vectors = test.get_vectors(image)
print(vectors)
```



### Example 15: Fetch Image from URL, Apply Noise Reduction, and Get Vectors

In this example, we will fetch an image from a URL, apply noise reduction to it with a kernel size of 11, and get the vectors for the entire image.

```
test = CLIPQ()
image = test.fetch_image_from_url(url="https://example.com/image.jpg")
blurred_image = test.apply_noise_reduction(image, kernel_size=11)
vectors = test.get_vectors(blurred_image)
print(vectors)
```


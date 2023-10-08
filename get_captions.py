from clipq.main import CLIPQ

test = CLIPQ()
candidates = [
    "a serene landscape",
    "a group of people",
    "an animal in the wild",
    "a building",
    "vehicles on the road",
]
image = test.fetch_image_from_url(url="https://picsum.photos/800")
best_caption = test.get_captions(image, candidate_captions=candidates)
print(best_caption)

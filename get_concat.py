from clipq.main import CLIPQ


test = CLIPQ()
candidates = [
    "a serene landscape",
    "a group of people",
    "an animal in the wild",
    "a building",
    "vehicles on the road",
]
image = test.load_image_from_path(path="agorabanner.png")
caption_result = test.get_and_concat_captions(
    image, candidate_captions=candidates, h_splits=2, v_splits=2
)
print(caption_result)

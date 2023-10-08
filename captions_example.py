from clipq.main import CLIPQ

test = CLIPQ(query_text="A photo of a cat")
captions = test.get_and_concat_captions(
    url="https://picsum.photos/800", h_splits=3, v_splits=3
)
print(captions)

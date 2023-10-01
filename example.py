from clipq.main import CLIPQ

# Usage
test = CLIPQ(query_text="A photo of a cat")
vectors = test.run_from_url(url="https://picsum.photos/800", h_splits=3, v_splits=3)
print(vectors)

from clipq.main import CLIPQ


test = CLIPQ()
image = test.fetch_image_from_url(url="https://picsum.photos/800")
image_embeds = test.embed_whole_image(image)
print(image_embeds)

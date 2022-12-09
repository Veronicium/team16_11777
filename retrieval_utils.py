from transformers import CLIPProcessor

preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def retrieval_collate_fn(data):
    images = [x[0] for x in data]
    captions = [x[1] for x in data]
    for x in data:
        captions.extend(x[2]) # share global negs
        images.extend(x[3])

    return preprocess(text=captions, images=images, return_tensors="pt", padding=True)

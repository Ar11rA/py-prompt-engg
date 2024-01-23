from urllib.request import urlopen

from PIL import Image
from sentence_transformers import SentenceTransformer, util

# Load CLIP model
model = SentenceTransformer('clip-ViT-B-32')

# Encode an image:
img_emb = model.encode(Image.open('../resources/product_images/sony_camera_front.jpg'))

# Encode queries - text or image
query_text_emb = model.encode(['Fridge', 'Camera', 'Laptop', 'Vegetable', 'Sony Camera', 'Sony Alpha Camera'])
query_img_emb_1 = model.encode(Image.open('../resources/product_images/sony_cam_side.jpg'))
query_img_emb_2 = model.encode(Image.open('../resources/product_images/sony_camera_front_2.jpg'))
query_img_emb_3 = model.encode(Image.open(urlopen(
    "https://upload.wikimedia.org/wikipedia/commons/b/be/Chinon_CP_9_AF_BW_1.JPG")
))
query_img_emb_4 = model.encode(Image.open(urlopen(
    "https://upload.wikimedia.org/wikipedia/commons/4/48/Canis_lupus_familiaris_Gda%C5%84sk.JPG")
))
query_img_emb_5 = model.encode(Image.open(urlopen(
    "https://upload.wikimedia.org/wikipedia/commons/6/68/Castl_Ned%C4%9Bli%C5%A1t%C4%9B_02.jpg")
))
query_img_emb_6 = model.encode(Image.open(urlopen(
    "https://upload.wikimedia.org/wikipedia/commons/b/b7/Lenovo_G500s_laptop-2903.jpg")
))

# Compute cosine similarities
print(util.cos_sim(img_emb, query_text_emb))
print(util.cos_sim(img_emb, query_img_emb_1))
print(util.cos_sim(img_emb, query_img_emb_2))
print(util.cos_sim(img_emb, query_img_emb_3))
print(util.cos_sim(img_emb, query_img_emb_4))
print(util.cos_sim(img_emb, query_img_emb_5))
print(util.cos_sim(img_emb, query_img_emb_6))

import os
import base64
import requests

# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")

BASE64_PREFIX = "data:image/jpeg;base64"


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}


payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
            "role": "system",
            "content": """ 
            Your job is to create product content for the given images
            """
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """
                            All the images are related to 1 product. Can you tell me more about the product?"
                            Product is called Sony Alpha ILCE-6400L
                            
                            Product Features:
                            
                            Megapixel: 80 mp
                            
                            Can you summarize the product description in 5 lines? Include megapixel value in the description.
                            If any product features are obtained from image, ignore them if they are present in the above context.
                            """
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"{BASE64_PREFIX},{encode_image('../resources/product_images/sony_camera_front.jpg')}",
                    },
                }, {
                    "type": "image_url",
                    "image_url": {
                        "url": f"{BASE64_PREFIX},{encode_image('../resources/product_images/sony_camera_front_2.jpg')}",
                    },
                }, {
                    "type": "image_url",
                    "image_url": {
                        "url": f"{BASE64_PREFIX},{encode_image('../resources/product_images/sony_lens.jpg')}",
                    },
                }, {
                    "type": "image_url",
                    "image_url": {
                        "url": f"{BASE64_PREFIX},{encode_image('../resources/product_images/sony_cam_back.jpg')}",
                    },
                }, {
                    "type": "image_url",
                    "image_url": {
                        "url": f"{BASE64_PREFIX},{encode_image('../resources/product_images/sony_cam_side.jpg')}",
                    },
                },
            ],
        }
    ],
    "max_tokens": 800,
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

print(response.json())

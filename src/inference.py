import torch
from Kimi_K25.modeling_kimi_k25 import KimiK25ForConditionalGeneration
from Kimi_K25.kimi_k25_processor import KimiK25Processor
from PIL import Image
import requests

class CuriousPI:
    def __init__(self, model_path):
        self.model = KimiK25ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.processor = KimiK25Processor.from_pretrained(model_path)
    
    @staticmethod
    def load(model_path):
        return CuriousPI(model_path)
    
    def generate_text(self, prompt, max_length=100):
        """Generate text response"""
        inputs = self.processor(text=prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=max_length)
        return self.processor.decode(outputs[0], skip_special_tokens=True)
    
    def generate_with_image(self, prompt, image_path, max_length=100):
        """Generate response with image"""
        image = Image.open(image_path)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=max_length)
        return self.processor.decode(outputs[0], skip_special_tokens=True)
    
    def generate_with_video(self, prompt, video_path, max_length=100):
        """Generate response with video"""
        # Video processing (requires additional setup)
        inputs = self.processor(text=prompt, videos=video_path, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=max_length)
        return self.processor.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Test
    model = CuriousPI.load("./models/curiosipi-v1")
    
    # Text only
    response = model.generate_text("What is artificial intelligence?")
    print(f"Response: {response}")

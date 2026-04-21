from src.inference import CuriousPI

model = CuriousPI.load("./models/curiosipi-v1")
response = model.generate("Hello, how are you?")
print(response)

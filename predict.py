import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate Model Architecture
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

# Load trained weights
model.load_state_dict(torch.load("model.pth", map_location=device))
model = model.to(device)
model.eval()

# Same Transform as Training
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Prediction Function
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    class_names = ["AI Generated", "Real Image"]

    label = class_names[predicted.item()]
    confidence = round(confidence.item() * 100, 2)

    return label, confidence
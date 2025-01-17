import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import segmentation_models_pytorch as smp  # For importing U-Net

# Define the model
class UNetWrapper(nn.Module):
    def __init__(self):
        super(UNetWrapper, self).__init__()
        self.unet = smp.Unet(
            encoder_name="resnet34",  # Use ResNet34 as the encoder
            encoder_weights="imagenet",  # Pretrained on ImageNet
            in_channels=1,  # For grayscale (MNIST-style)
            classes=1,  # Binary segmentation output
        )

    def forward(self, x):
        return self.unet(x)

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Testing function
def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets.unsqueeze(1).float())
            total_loss += loss.item()
            predictions = torch.sigmoid(outputs).round()
            correct += (predictions == targets.unsqueeze(1)).sum().item()
    accuracy = correct / (len(test_loader.dataset))
    return total_loss / len(test_loader), accuracy

# Federated client class
class Client(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_loss = train(self.model, self.train_loader, self.criterion, self.optimizer, self.device)
        return self.get_parameters(), len(self.train_loader.dataset), {"train_loss": train_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.test_loader, self.criterion, self.device)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

# Main function
if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformations and dataset loading
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    target_transform = transforms.Compose([transforms.Lambda(lambda y: torch.tensor(y, dtype=torch.float32))])

    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform, target_transform=target_transform
    )
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Model initialization
    model = UNetWrapper().to(device)

    # Start the Flower client
    client = Client(model, train_loader, test_loader, device)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
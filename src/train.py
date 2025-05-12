# Training Loop

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data import get_dataloaders
from network import DeepLabV3Plus
import globals as G

def train_model(train_loader, val_loader, train_decoder_only=True):
    model = DeepLabV3Plus().to(G.DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(
        model.parameters() if not train_decoder_only else model.model.classifier.parameters(),
        lr=G.LEARNING_RATE
    )

    for epoch in range(G.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{G.NUM_EPOCHS}]")

        for images, masks in loop:
            images = images.to(G.DEVICE)
            masks = masks.to(G.DEVICE)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} loss: {running_loss / len(train_loader):.4f}")

        # Optional: save checkpoint
        torch.save(model.state_dict(), f"{G.SAVE_DIR}/model_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders()
    train_model(train_loader, val_loader, train_decoder_only=True)

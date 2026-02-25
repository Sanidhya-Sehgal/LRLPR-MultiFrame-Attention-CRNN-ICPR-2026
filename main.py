import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split

############################################
# Dataset
############################################

class TrackDataset(Dataset):

    def __init__(self, tracks, char2idx):
        self.tracks = tracks
        self.char2idx = char2idx

        self.transform = transforms.Compose([
            transforms.Resize((32,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def encode(self, text):
        return torch.tensor(
            [self.char2idx[c] for c in text],
            dtype=torch.long
        )

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):

        item = self.tracks[idx]
        imgs = []

        for p in item["lr_paths"]:
            img = Image.open(p).convert("RGB")
            imgs.append(self.transform(img))

        imgs = torch.stack(imgs)
        label = self.encode(item["label"])

        return imgs, label


def collate_fn(batch):

    images, labels = zip(*batch)

    images = torch.stack(images)
    label_lengths = torch.tensor([len(l) for l in labels])
    labels = torch.cat(labels)

    return images, labels, label_lengths


############################################
# Model
############################################

class MultiFrameCRNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        self.cnn = nn.Sequential(
            *list(backbone.children())[:-2]
        )

        self.attn = nn.Sequential(
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

        self.pool = nn.AdaptiveAvgPool2d((1,None))

        self.lstm = nn.LSTM(
            512,
            256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self,x):

        B,T,C,H,W = x.size()

        x = x.view(B*T,C,H,W)
        f = self.cnn(x)

        _,F,Hf,Wf = f.size()
        f = f.view(B,T,F,Hf,Wf)

        frame_feat = f.mean([3,4])
        w = torch.softmax(self.attn(frame_feat),dim=1)

        f = (f*w.unsqueeze(-1).unsqueeze(-1)).sum(1)

        f = self.pool(f).squeeze(2)
        f = f.permute(0,2,1)

        seq,_ = self.lstm(f)
        logits = self.fc(seq)

        return logits.permute(1,0,2)


############################################
# Decode
############################################

def greedy_decode(logits, idx2char):

    preds = torch.argmax(
        F.softmax(logits,2),
        2
    ).permute(1,0)

    results=[]

    for seq in preds:
        prev=-1
        text=""

        for i in seq:
            i=i.item()
            if i!=0 and i!=prev:
                text+=idx2char[i]
            prev=i

        results.append(text[:7])

    return results


############################################
# Training
############################################

def train(model,loader,optimizer,ctc,device):

    model.train()
    total=0

    for imgs,labels,ll in loader:

        imgs=imgs.to(device)
        labels=labels.to(device)
        ll=ll.to(device)

        optimizer.zero_grad()

        logits=model(imgs)
        log_probs=F.log_softmax(logits,2)

        inp_len=torch.full(
            (logits.size(1),),
            logits.size(0),
            dtype=torch.long
        ).to(device)

        loss=ctc(
            log_probs,
            labels,
            inp_len,
            ll
        )

        loss.backward()
        optimizer.step()

        total+=loss.item()

    return total/len(loader)


############################################
# MAIN
############################################

def main():

    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",default="train")
    args=parser.parse_args()

    device=torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:",device)

    # Dummy char set example
    CHARS="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    char2idx={c:i+1 for i,c in enumerate(CHARS)}
    idx2char={i:c for c,i in char2idx.items()}

    num_classes=len(CHARS)+1

    model=MultiFrameCRNN(num_classes).to(device)

    if args.mode=="train":
        print("Training pipeline ready.")

    if args.mode=="test":
        print("Inference pipeline ready.")


if __name__=="__main__":
    main()
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from datasets import load_dataset
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np

dataset = load_dataset("glue", "sst2")

def swiglu(x):
    x1, x2 = x.chunk(2, dim=-1)
    return F.silu(x1)*x2

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model) 
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        qkv = self.qkv_proj(x) 
        qkv = qkv.reshape(B, T, self.num_heads, 3 * self.d_k).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().reshape(B, T, C)

        return self.out_proj(context)
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, activation="relu"):
        super().__init__()
        self.activation = activation
        if activation == "swiglu":
            self.linear = nn.Linear(d_model, d_ff * 2)
            self.output = nn.Linear(d_ff, d_model)
        else:
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
        self.last_pre_activation = None
        self.last_post_activation = None

    def forward(self, x):
        if self.activation == "swiglu":
            x = self.linear(x)
            x1, x2 = x.chunk(2, dim=-1)
            self.last_pre_activation = x1.detach().cpu()
            x = F.silu(x1) * x2
            x = self.output(x)
        elif self.activation == "swish":
            x = self.linear1(x)
            self.last_pre_activation = x.detach().cpu()
            x = F.silu(x)
            x = self.linear2(x)
        else:  # relu
            x = self.linear1(x)
            self.last_pre_activation = x.detach().cpu()
            x = F.relu(x)
            x = self.linear2(x)
        self.last_post_activation = x.detach().cpu()
        return x
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, activation="relu"):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len=512, activation="relu"):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, activation)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerForClassification(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, num_classes, max_len=512, activation="relu"):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len, activation)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        enc_out = self.encoder(input_ids)
        cls_token = enc_out[:, 0, :]
        logits = self.classifier(cls_token)
        return logits

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size
num_classes = 2
test_relu = TransformerForClassification(vocab_size, 64, 2, 128, 1, num_classes=2, activation="relu")
test_swish = TransformerForClassification(vocab_size, 64, 2, 128, 1, num_classes=2, activation="swish")
test_swiglu = TransformerForClassification(vocab_size, 64, 2, 128, 1, num_classes=2, activation="swiglu")

# 간단한 전처리 함수 정의 (CLS 토큰 명시적 추가)
def preprocess(batch):
    tokenized = tokenizer(["[CLS] " + s for s in batch["sentence"]], padding="max_length", truncation=True, max_length=32)
    return {
        "input_ids": tokenized["input_ids"],
        "labels": batch["label"]
    }

# 데이터셋 준비
train_data = dataset["train"].select(range(3000)).map(preprocess, batched=True)
val_data = dataset["validation"].select(range(500)).map(preprocess, batched=True)

train_data.set_format(type="torch")
val_data.set_format(type="torch")

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 평가 함수
def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids)
            pred = logits.argmax(dim=-1).cpu().numpy()
            preds.extend(pred)
            targets.extend(labels.cpu().numpy())
    return accuracy_score(targets, preds)

def train_model(model, name, loss_log):
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    accuracies = []
    grad_norms = []
    for epoch in range(10):
        epoch_losses = []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids)
            loss = loss_fn(logits, labels)
            loss.backward()
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)
            optimizer.step()
            optimizer.zero_grad()
            epoch_losses.append(loss.item())
        avg_loss = np.mean(epoch_losses)
        loss_log.append(avg_loss)
        acc = evaluate(model, val_loader)
        accuracies.append(acc)
        total_norm = sum(p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        grad_norms.append(total_norm)
        print(f"[{name}] Epoch {epoch+1} Loss: {avg_loss:.4f} Accuracy: {acc:.4f}")
    return accuracies, grad_norms

# loss log 저장용 리스트
relu_loss_log, swish_loss_log, swiglu_loss_log = [], [], []

# 학습 및 로그 저장
relu_accuracies, relu_grads = train_model(test_relu, "ReLU", relu_loss_log)
swish_accuracies, swish_grads = train_model(test_swish, "Swish", swish_loss_log)
swiglu_accuracies, swiglu_grads = train_model(test_swiglu, "SwiGLU", swiglu_loss_log)

relu_post = test_relu.encoder.layers[-1].ffn.last_post_activation.flatten().numpy()
swish_post = test_swish.encoder.layers[-1].ffn.last_post_activation.flatten().numpy()
swiglu_post = test_swiglu.encoder.layers[-1].ffn.last_post_activation.flatten().numpy()

plt.figure()
plt.hist(relu_post, bins=50, alpha=0.6, label="ReLU post-activation")
plt.hist(swish_post, bins=50, alpha=0.6, label="Swish post-activation")
plt.hist(swiglu_post, bins=50, alpha=0.6, label="SwiGLU post-activation")
plt.xlabel("Post-activation value")
plt.ylabel("Frequency")
plt.title("Distribution of Post-activation Values")
plt.legend()
plt.tight_layout()
plt.savefig("post_activation_histogram.png")

# Loss curves
plt.figure()
plt.plot(relu_loss_log, label='ReLU Loss')
plt.plot(swish_loss_log, label='Swish Loss')
plt.plot(swiglu_loss_log, label='SwiGLU Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")

# Accuracy curves
plt.figure()
plt.plot(relu_accuracies, label='ReLU Acc')
plt.plot(swish_accuracies, label='Swish Acc')
plt.plot(swiglu_accuracies, label='SwiGLU Acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy per Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_curve.png")

# Gradient norm curves
plt.figure()
plt.plot(relu_grads, label='ReLU Grad Norm')
plt.xlabel("Epoch")
plt.ylabel("Gradient Norm (L2)")
plt.title("ReLU Gradient Flow per Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("relu_gradient_flow.png")

plt.figure()
plt.plot(swish_grads, label='Swish Grad Norm')
plt.xlabel("Epoch")
plt.ylabel("Gradient Norm (L2)")
plt.title("Swish Gradient Flow per Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("swish_gradient_flow.png")

plt.figure()
plt.plot(swiglu_grads, label='SwiGLU Grad Norm')
plt.xlabel("Epoch")
plt.ylabel("Gradient Norm (L2)")
plt.title("SwiGLU Gradient Flow per Epoch")
plt.legend()
plt.tight_layout()
plt.savefig("swiglu_gradient_flow.png")

print("📊그래프 저장완료.")
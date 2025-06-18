import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from einops import rearrange
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torch.utils.data import Dataset, DataLoader

# ==================== 模型部分 ====================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, mlp_dim=1024, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, dim_head)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class TransUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        in_ch = in_channels
        for feature in features:
            self.encoder.append(ConvBlock(in_ch, feature))
            in_ch = feature

        # Transformer
        self.transformer = nn.Sequential(
            TransformerBlock(features[-1], heads=4, dim_head=32)
        )

        # Decoder
        for feature in reversed(features[:-1]):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(ConvBlock(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for enc in self.encoder[:-1]:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Last encoder block
        x = self.encoder[-1](x)

        # Transformer
        x_transformed = rearrange(x, 'b c h w -> b (h w) c')
        x_transformed = self.transformer(x_transformed)
        x = rearrange(x_transformed, 'b (h w) c -> b c h w', h=x.shape[2])

        # Decoder
        skip_connections = skip_connections[::-1]  # reverse
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip = skip_connections[idx // 2]

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            x = torch.cat((skip, x), dim=1)
            x = self.decoder[idx + 1](x)

        return torch.sigmoid(self.final_conv(x))

# ==================== 数据集部分 ====================

class MoNuSegDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        参数:
            root_dir (string): 数据集根目录
            split (string): 训练集/验证集/测试集 ('train', 'val', 'test')
            transform: 数据增强转换
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # 获取图像文件列表
        if split == 'train':
            self.img_dir = os.path.join(root_dir, 'Training', 'images')
            self.mask_dir = os.path.join(root_dir, 'Training', 'masks')
        else:
            self.img_dir = os.path.join(root_dir, 'Test', 'images')
            self.mask_dir = os.path.join(root_dir, 'Test', 'masks')

        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.tif')])

        # 划分训练集、验证集和测试集
        if split == 'train':
            self.img_files = self.img_files[:int(len(self.img_files) * 0.6)]  # 60%用于训练
        elif split == 'val':
            self.img_files = self.img_files[int(len(self.img_files) * 0.6):int(len(self.img_files) * 0.8)]  # 20%用于验证
        else:  # test
            self.img_files = self.img_files[int(len(self.img_files) * 0.8):]  # 20%用于测试

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 加载图像
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.tif', '.png'))

        # 读取图像和掩码
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 调整图像大小为256x256
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        # 标准化掩码为0-1
        mask = mask / 255.0

        # 应用数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.unsqueeze(0)  # 添加channel维度


def get_transforms(split):
    if split == 'train':
        return A.Compose([
            A.RandomCrop(width=256, height=256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussNoise(p=0.2),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.GaussianBlur(p=0.5),
                A.MotionBlur(p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
            ], p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def get_dataloaders(root_dir, batch_size=8, num_workers=4):
    train_dataset = MoNuSegDataset(root_dir, split='train', transform=get_transforms('train'))
    val_dataset = MoNuSegDataset(root_dir, split='val', transform=get_transforms('val'))
    test_dataset = MoNuSegDataset(root_dir, split='test', transform=get_transforms('test'))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

# ==================== 训练和评估部分 ====================

def dice_coefficient(pred, target):
    smooth = 1e-5
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    return loss.mean()


def iou_score(pred, target):
    smooth = 1e-5
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    union = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) - intersection
    return ((intersection + smooth) / (union + smooth)).mean()


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        return 1 - dice_coefficient(pred, target)


def train_model(model, train_loader, val_loader, num_epochs=100, device='cuda'):
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_dice = 0
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}') as pbar:
            for images, masks in pbar:
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                # 计算Dice系数
                dice = dice_coefficient(outputs, masks)
                epoch_loss += loss.item()
                epoch_dice += dice.item()
                pbar.set_postfix({'loss': loss.item(), 'dice': dice.item()})

        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_dice = epoch_dice / len(train_loader)
        train_losses.append(avg_train_loss)
        train_dices.append(avg_train_dice)

        # 验证
        model.eval()
        val_loss = 0
        val_dice = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                dice = dice_coefficient(outputs, masks)
                val_loss += loss.item()
                val_dice += dice.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        val_losses.append(avg_val_loss)
        val_dices.append(avg_val_dice)

        print(f'Epoch {epoch + 1}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        scheduler.step()

    # 绘制损失曲线和Dice系数曲线
    plt.figure(figsize=(15, 5))

    plt.subplot(121)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(122)
    plt.plot(train_dices, label='Train Dice')
    plt.plot(val_dices, label='Validation Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.title('Training and Validation Dice Coefficient')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

    return train_losses, val_losses


def save_prediction(image, mask, pred, filename):
    image = image.cpu().numpy().transpose(1, 2, 0)
    mask = mask.cpu().numpy()[0]
    pred = pred.cpu().numpy()[0]

    # 归一化图像用于显示
    image = (image - image.min()) / (image.max() - image.min())

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(mask, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')

    # 计算当前样本的Dice系数
    current_dice = dice_coefficient(
        torch.tensor(pred).unsqueeze(0).unsqueeze(0),
        torch.tensor(mask).unsqueeze(0).unsqueeze(0)
    ).item()

    plt.subplot(133)
    plt.imshow(pred, cmap='gray')
    plt.title(f'Prediction (Dice={current_dice:.4f})')
    plt.axis('off')

    plt.suptitle(f'Sample: {filename.split("_")[-1].split(".")[0]}', y=1.05)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()


def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    dice_scores = []
    iou_scores = []

    # 创建保存结果的目录
    os.makedirs('test_results', exist_ok=True)

    # 用于记录总样本数
    total_samples = 0

    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)

            # 计算每个样本的Dice系数
            for i in range(images.size(0)):
                dice = dice_coefficient(outputs[i:i + 1], masks[i:i + 1])
                iou = iou_score(outputs[i:i + 1], masks[i:i + 1])

                dice_scores.append(dice.item())
                iou_scores.append(iou.item())

                # 保存每个样本的预测结果
                save_prediction(
                    images[i],
                    masks[i],
                    outputs[i],
                    f'test_results/prediction_{total_samples + i:03d}.png'
                )

            total_samples += images.size(0)

    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)
    std_dice = np.std(dice_scores)

    print(f'\nTest Results:')
    print(f'Total test samples: {total_samples}')
    print(f'Mean Dice Score: {mean_dice:.4f} ± {std_dice:.4f}')
    print(f'Mean IoU Score: {mean_iou:.4f}')

    # 绘制Dice分布直方图
    plt.figure(figsize=(10, 5))
    plt.hist(dice_scores, bins=20, edgecolor='black')
    plt.xlabel('Dice Coefficient')
    plt.ylabel('Count')
    plt.title(f'Distribution of Dice Coefficients on Test Set (n={total_samples})')
    plt.savefig('test_results/dice_distribution.png')
    plt.close()

    # 保存详细的评估结果到文本文件
    with open('test_results/evaluation_results.txt', 'w') as f:
        f.write(f'Total test samples: {total_samples}\n')
        f.write(f'Mean Dice Score: {mean_dice:.4f} ± {std_dice:.4f}\n')
        f.write(f'Mean IoU Score: {mean_iou:.4f}\n')
        f.write('\nDetailed Dice scores:\n')
        for i, score in enumerate(dice_scores):
            f.write(f'Sample {i:03d}: {score:.4f}\n')

    return mean_dice, mean_iou

# ==================== 主函数 ====================

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 获取数据加载器
    train_loader, val_loader, test_loader = get_dataloaders('MoNuSeg3', batch_size=2)

    # 创建模型
    model = TransUNet().to(device)

    # 训练模型
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=25, device=device)

    # 加载最佳模型进行测试
    model.load_state_dict(torch.load('best_model.pth'))
    mean_dice, mean_iou = evaluate_model(model, test_loader, device=device)


if __name__ == '__main__':
    main()
import albumentations as A
import argparse
import cv2
import numpy as np
import os
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

# 🔥 引入指标计算
from sklearn.metrics import classification_report

from PIL import Image
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from typing import Tuple

# 设置设备 (优先使用 GPU)
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="字体分类训练脚本")

    # 添加参数
    parser.add_argument(
        "--image_folder",
        type=str,
        default="/root/autodl-tmp/font-classify/dataset_fonts",
        help="包含图片数据的文件夹路径",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="/root/autodl-tmp/font-classify/model_all", # 🔥 改了名字，避免覆盖旧模型
        help="训练好的模型保存路径",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.15,
        help="用于测试（验证）的数据集比例 (例如 0.15 表示 15%)",
    )
    parser.add_argument(
        "-net",
        "--network_type",
        type=str,
        default="resnet50",
        help="使用的网络架构类型 (例如 resnet50)",
    )
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="批处理大小 (Batch size)")
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=0.0001, help="初始学习率"
    )
    parser.add_argument(
        "-e", "--num_epochs", type=int, default=100, help="训练总轮数 (Epochs)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="数据加载器的工作线程数"
    )

    # 解析参数
    args = parser.parse_args()

    return args


class CustomImageFolder(ImageFolder):
    """自定义图像文件夹加载器，支持 Albumentations 增强"""
    def __init__(self, root, transform=None, **kwargs):
        super(CustomImageFolder, self).__init__(root, **kwargs)
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path).convert("RGB") # 确保转为 RGB 防止单通道报错

        if self.transform is not None:
            sample = np.array(sample)  # 将 PIL 图片转为 Numpy 数组
            transformed = self.transform(image=sample)  # 应用增强
            sample = transformed["image"]  # 提取增强后的图片

        return sample, target


class ResizeWithPad:
    """保持纵横比缩放并填充背景"""
    def __init__(
        self, new_shape: Tuple[int, int], padding_color: Tuple[int] = (255, 255, 255)
    ) -> None:
        self.new_shape = new_shape
        self.padding_color = padding_color

    def __call__(self, image: np.array, **kwargs) -> np.array:
        original_shape = (image.shape[1], image.shape[0])
        ratio = float(max(self.new_shape)) / max(original_shape)
        new_size = tuple([int(x * ratio) for x in original_shape])
        image = cv2.resize(image, new_size)
        delta_w = self.new_shape[0] - new_size[0]
        delta_h = self.new_shape[1] - new_size[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=self.padding_color,
        )
        return image


class CutMax:
    """如果图片超过最大尺寸，则进行裁剪"""

    def __init__(self, max_size: int = 1024) -> None:
        self.max_size = max_size

    def __call__(self, image: np.array, **kwargs) -> np.array:
        if image.shape[0] > self.max_size:
            image = image[: self.max_size, :, :]
        if image.shape[1] > self.max_size:
            image = image[:, : self.max_size, :]
        return image


def main(args):
    os.makedirs(args.output_folder, exist_ok=True)

    # ============================================================
    # 🛠️ 关键修改：优化后的数据增强策略
    # ============================================================
    print("🔧 正在应用优化后的数据增强策略 (已修复隶书/幼圆识别问题)...")
    transform = A.Compose(
        [
            A.Lambda(image=CutMax(1024)),
            A.Lambda(image=ResizeWithPad((320, 320))),  # 自定义方形填充
            
            # 🔥 修改 1：针对隶书，大幅降低旋转角度 (60 -> 10)
            A.ShiftScaleRotate(
                shift_limit=0.1,        # 稍微平移
                scale_limit=(0.9, 1.1), # 缩放幅度减小
                rotate_limit=10,        # 关键！保护隶书结构不被破坏
                interpolation=1,
                p=0.5,
            ),
            
            # 颜色增强保留，增加模型对光照的鲁棒性
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.3),
            
            # 🔥 修改 2：移除了 ISONoise 和 ImageCompression
            # 删除了这两行，保护幼圆的清晰度，防止圆角变糊被误判为黑体
            # A.ISONoise(p=0.2), 
            # A.ImageCompression(quality_lower=70, quality_upper=95, p=0.2),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    # 预览用的 Transform (无人眼不可见的归一化)
    check_transform = A.Compose(
        [
            A.Lambda(image=CutMax(1024)),
            A.Lambda(image=ResizeWithPad((320, 320))),
            
            # 🔥 这里参数与训练保持一致 (10度)，确保预览图真实反映训练情况
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=(0.9, 1.1), 
                rotate_limit=10, 
                interpolation=1, 
                p=0.5
            ),
            
            # 颜色增强预览
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.3),
        ]
    )

    image_folder = args.image_folder
    network_type = args.network_type
    best_model_params_path = os.path.join(args.output_folder, "best_model_params.pt")

    # 数据集设置
    dataset = CustomImageFolder(image_folder, transform=transform)
    
    # 自动识别剩下的类别
    class_names = dataset.classes
    print(f"✅ 成功检测到 {len(class_names)} 个分类")
    
    n = len(dataset)
    n_test = int(args.test_split * n)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [n - n_test, n_test]
    )

    # 保存增强后的检查图片 (Check images)
    check_dataset = CustomImageFolder(image_folder, transform=check_transform)
    Path(os.path.join(args.output_folder, "check")).mkdir(parents=True, exist_ok=True)
    print("💾 正在保存增强效果预览图 (Check images)...")
    for i in range(min(20, len(check_dataset))): # 限制保存 20 张以节省时间
        img_data = check_dataset[i]
        img = img_data[0] # 获取图片部分
        Image.fromarray(img).save(os.path.join(args.output_folder, "check", f"{i}.png"))

    # 保存类别名称列表
    with open(os.path.join(args.output_folder, "class_names.txt"), "w") as f:
        for item in class_names:
            f.write(f"{item}\n")

    dataset_sizes = {"train": len(train_dataset), "val": len(test_dataset)}

    # 数据加载器 (Dataloaders)
    batch_size = args.batch_size
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, num_workers=args.num_workers, batch_size=batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, num_workers=args.num_workers, batch_size=batch_size, shuffle=True
    )
    dataloaders = {"train": train_dataloader, "val": test_dataloader}

    # 模型创建
    print(f"🏗️ 正在创建模型架构: {network_type}")
    model = timm.create_model(
        network_type, pretrained=True, num_classes=len(class_names)
    )
    model.to(device)

    # 损失函数 & 优化器
    # 🔥 修改 3：加入 label_smoothing (标签平滑) 防止模型过于自信，缓解相似字体混淆
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.num_epochs, T_mult=1, eta_min=0
    )

    writer = SummaryWriter(log_dir=os.path.join(args.output_folder, "runs"))

    # 训练循环
    best_acc = 0.0

    for epoch in range(args.num_epochs):
        print(f"\n轮次 (Epoch) {epoch}/{args.num_epochs - 1}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            # 用于计算详细指标的容器
            val_preds = []
            val_labels = []

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} 阶段"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # 如果是验证阶段，收集数据用于生成分类报告
                if phase == "val":
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} 损失(Loss): {epoch_loss:.4f} 准确率(Acc): {epoch_acc:.4f}")

            # TensorBoard 日志记录
            writer.add_scalar(f"{phase}/Loss", epoch_loss, epoch)
            writer.add_scalar(f"{phase}/Accuracy", epoch_acc, epoch)

            # 详细的每类指标记录 (仅在验证阶段 Val)
            if phase == "val":
                print("\n📊 分类详情报告 (Classification Report):")
                # 1. 打印人类可读的表格报告
                print(classification_report(val_labels, val_preds, target_names=class_names, digits=4))
                
                # 2. 获取字典格式以便写入 TensorBoard
                report_dict = classification_report(val_labels, val_preds, target_names=class_names, output_dict=True)
                
                # 3. 写入 TensorBoard
                for cls_name in class_names:
                    if cls_name in report_dict:
                        writer.add_scalar(f"Class_F1/{cls_name}", report_dict[cls_name]['f1-score'], epoch)
                        writer.add_scalar(f"Class_Precision/{cls_name}", report_dict[cls_name]['precision'], epoch)
                        writer.add_scalar(f"Class_Recall/{cls_name}", report_dict[cls_name]['recall'], epoch)
                
                writer.add_scalar("Overall/Macro_F1", report_dict['macro avg']['f1-score'], epoch)

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

        print(f"当前最佳验证准确率: {best_acc:.4f}")

    # 加载最佳模型并保存最终版本
    model.load_state_dict(torch.load(best_model_params_path))
    torch.save(
        model.state_dict(), os.path.join(args.output_folder, "trained_model.pth")
    )

    writer.close()
    print("🚀 训练全部完成！")


if __name__ == "__main__":
    args = parse_args()
    main(args)
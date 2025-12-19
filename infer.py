import albumentations as A
import argparse
import numpy as np
import os
import timm
import torch
import csv  # 必须导入 csv 库

from albumentations.pytorch import ToTensorV2
from train import CutMax, ResizeWithPad
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("--model_folder", type=str, default="/root/autodl-tmp/font-classify/model_all", help="Path where the trained model was saved")
    parser.add_argument("--data_folder", type=str, default="/root/autodl-tmp/font-classify/sample_data/test", help="Path to images to run inference on")
    parser.add_argument("-net", "--network_type", type=str, default="resnet50", help="Type of network architecture")
    parser.add_argument("--output_file", type=str, default="inference_results.csv", help="Filename to save the results")
    args = parser.parse_args()
    return args


def main(args):
    # 1. 加载类别
    with open(os.path.join(args.model_folder, "class_names.txt"), "r") as f:
        class_names = f.read().splitlines()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 加载模型
    model = timm.create_model(args.network_type, pretrained=False, num_classes=len(class_names))
    model.to(device)

    model_path = os.path.join(args.model_folder, "best_model_params.pt")
    # model_path = os.path.join(args.model_folder, "trained_model.pth")
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint)
    model.eval()

    # 3. 定义预处理
    transform = A.Compose([
        # A.Lambda(image=CutMax(1024)),
        A.Lambda(image=ResizeWithPad((320, 320))),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # 4. 准备保存路径
    save_path = os.path.join(args.model_folder, args.output_file)
    print(f"📄 结果将保存到: {save_path}")

    # 🔥🔥🔥 核心修正：先打开文件，再开始循环 🔥🔥🔥
    with open(save_path, mode='w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.writer(csv_file)
        # 写入表头
        writer.writerow(['Filename', 'Prediction', 'Confidence'])

        # 5. 开始循环预测
        for image_file in os.listdir(args.data_folder):
            image_path = os.path.join(args.data_folder, image_file)
            
            # [安全修正] 增加异常处理和 RGB 转换
            try:
                image_pil = Image.open(image_path).convert("RGB")
                image = np.array(image_pil)
            except Exception as e:
                print(f"Skipping {image_file}: {e}")
                continue

            # 预处理
            image_tensor = transform(image=image)["image"].unsqueeze(0)
            image_tensor = image_tensor.to(device)
            
            # 推理
            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, prediction = torch.max(probs, 1)
            
            # 获取结果
            class_name = class_names[prediction.item()]
            conf_score = confidence.item()
            
            # 打印到控制台
            print(f"{image_file:<30} {class_name:<20} {conf_score:.2%}")

            # 🔥 写入 CSV (现在 writer 已经定义了，可以用了)
            writer.writerow([image_file, class_name, f"{conf_score:.4f}"])

    print("Done!")

if __name__ == "__main__":
    args = parse_args()
    main(args)
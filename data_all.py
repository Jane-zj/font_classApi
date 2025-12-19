"""
字体分类数据集生成器 (全量字体版 - 固定数量)
流程：
1. 扫描：加载目录下所有字体文件 (无白名单限制)。
2. 生成：每个字体固定生成 200 张 (Times用英文，其他用中文)。
3. 绘制：在 800x800 大画布上绘制。
4. 裁剪：紧贴文字裁剪掉多余背景 (保留 PADDING)。
5. 输出：直接保存裁剪后的原图 (尺寸不固定)。
"""

import colorsys
import cv2
import numpy as np
import os
import sys
import random
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from pathlib import Path

# ================= ⚙️ 全局配置区域 ⚙️ =================

# 1. 任务量配置
IMAGES_PER_FONT = 200        # ✅ 修改：每个字体固定生成 200 张
CANVAS_SIZE = (800, 800)     # 临时画布 (足够大即可)

# 2. 核心算法参数
BACKGROUND_RATIO = 0.5       # 50% 使用图片背景
CONTRAST_COLOR_RATIO = 0.8   # 80% 保证高对比度
FONT_SIZE_RANGE = (50, 120)  # 字号范围
PADDING = 2                  # 裁剪时保留的边缘 (像素)

# 3. 路径配置
FONTS_DIR = "/root/autodl-tmp/font-classify/fonts"
BG_DIR = "/root/autodl-tmp/font-classify/sample_data/backgrounds/"
CORPUS_FILE = "/root/autodl-tmp/font-classify/card_corpus.txt"
OUTPUT_DIR = "/root/autodl-tmp/font-classify/dataset_fonts"

# 4. 字体白名单 (已移除，自动扫描所有字体)

# =======================================================

def ensure_corpus_exists():
    if os.path.exists(CORPUS_FILE): return
    os.makedirs(os.path.dirname(CORPUS_FILE), exist_ok=True)
    lines = ["张三 13800138000 经理", "李四 科技有限公司", "王五 设计总监"] * 1000
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def get_pastel_color():
    """生成浅色背景"""
    h = random.random()
    s = random.uniform(0.1, 0.5) 
    l = random.uniform(0.7, 0.95) 
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (int(r*255), int(g*255), int(b*255))

def rgb_to_hls(rgb): return colorsys.rgb_to_hls(*[x / 255.0 for x in rgb])
def hls_to_rgb(hls): return tuple([int(x * 255) for x in colorsys.hls_to_rgb(*hls)])
def opposite_color_hls(rgb):
    h, l, s = rgb_to_hls(rgb)
    return hls_to_rgb(((h + 0.5) % 1, max(0.2, 1.0 - l), s))

class ResizeWithPad:
    def __init__(self, new_shape, padding_color=(255, 255, 255)):
        self.new_shape = new_shape
        self.padding_color = padding_color
    def __call__(self, image):
        original_shape = (image.shape[1], image.shape[0])
        ratio = float(max(self.new_shape)) / max(original_shape)
        new_size = tuple([int(x * ratio) for x in original_shape])
        image = cv2.resize(image, new_size)
        delta_w, delta_h = self.new_shape[0] - new_size[0], self.new_shape[1] - new_size[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.padding_color)

class FontGenerator:
    def __init__(self):
        self.fonts = {}
        self.backgrounds = []
        self.fonts_cache = {}
        self.bg_resizer = ResizeWithPad(CANVAS_SIZE)
        
        if os.path.exists(BG_DIR):
            for f in os.listdir(BG_DIR):
                if f.lower().endswith((".jpg", ".png", ".jpeg")): 
                    self.backgrounds.append(os.path.join(BG_DIR, f))
        
        ensure_corpus_exists()
        with open(CORPUS_FILE, "r", encoding="utf-8") as f:
            self.chinese_lines = [line.strip() for line in f if len(line.strip()) > 3]
            if not self.chinese_lines: self.chinese_lines = ["样本不足"]

        self.english_lines = [
            "Tel: +86 138 0000 0000", "Mobile: 139-1234-5678", "Fax: 010-88888888",
            "www.google.com", "CEO & Founder", "205800091270", "575918", "No. 888888",
            "VIP: 88888888", "Copyright (C) 2025", "Design Studio", "Creative Group"
        ]
        for _ in range(100):
            self.english_lines.append("".join([str(random.randint(0,9)) for _ in range(random.randint(6, 15))]))

        print(f"🔍 正在全量扫描字体目录: {FONTS_DIR}")
        loaded_count = 0
        for root, _, files in os.walk(FONTS_DIR):
            for file in files:
                # ✅ 修改：只要是字体文件就加载，不再检查白名单
                if file.lower().endswith((".ttf", ".otf", ".woff", ".woff2")):
                    name = os.path.splitext(file)[0]
                    self.fonts[name] = os.path.join(root, file)
                    # print(f"   ✅ 加载: {name}") # 如果字体太多，可以注释掉这行减少刷屏
                    loaded_count += 1
        
        if loaded_count == 0:
            print("❌ 错误: 目录中未找到任何字体文件！")
            sys.exit(1)
        print(f"📦 共加载 {loaded_count} 个字体文件")

    def get_font(self, name, size):
        key = f"{name}_{size}"
        if key in self.fonts_cache: return self.fonts_cache[key]
        try:
            font = ImageFont.truetype(self.fonts[name], size)
        except:
            # Fallback (may fail if default not available, but usually fine)
            font = ImageFont.truetype(self.fonts[name], size)
        self.fonts_cache[key] = font
        return font

    def get_random_background(self):
        if not self.backgrounds: return None
        bg_path = random.choice(self.backgrounds)
        try:
            image = Image.open(bg_path).convert("RGB")
            if image.width > CANVAS_SIZE[0] and image.height > CANVAS_SIZE[1]:
                x = random.randint(0, image.width - CANVAS_SIZE[0])
                y = random.randint(0, image.height - CANVAS_SIZE[1])
                image = image.crop((x, y, x + CANVAS_SIZE[0], y + CANVAS_SIZE[1]))
            else:
                image = Image.fromarray(self.bg_resizer(np.array(image)))
            return image
        except:
            return None

    def generate(self, font_name, output_path):
        # 简单判断是否用英文 (Times系列)
        if "Times" in font_name:
            text = random.choice(self.english_lines)
        else:
            text = random.choice(self.chinese_lines)
            if len(text) > 15:
                 start = random.randint(0, len(text)-10)
                 text = text[start:start+random.randint(5, 15)]

        font_size = random.randint(*FONT_SIZE_RANGE)
        
        # 背景
        image = None
        if self.backgrounds and random.random() < BACKGROUND_RATIO:
            image = self.get_random_background()
            if not image: image = Image.new("RGB", CANVAS_SIZE, get_pastel_color())
        else:
            image = Image.new("RGB", CANVAS_SIZE, get_pastel_color())
        
        # 颜色
        font_color = (0, 0, 0)
        bg_sample = image.getpixel((CANVAS_SIZE[0]//2, CANVAS_SIZE[1]//2))
        if random.random() < CONTRAST_COLOR_RATIO:
            avg_bg = sum(bg_sample)/3
            font_color = (0,0,0) if avg_bg > 100 else (255,255,255)
            if random.random() < 0.2:
                 c = opposite_color_hls(bg_sample)
                 font_color = c

        draw = ImageDraw.Draw(image)
        font = self.get_font(font_name, font_size)
        bbox = font.getbbox(text)
        text_w, text_h = bbox[2], bbox[3]
        
        max_x = max(PADDING, CANVAS_SIZE[0] - text_w - PADDING)
        max_y = max(PADDING, CANVAS_SIZE[1] - text_h - PADDING)
        x = random.randint(PADDING, max_x) if max_x > PADDING else PADDING
        y = random.randint(PADDING, max_y) if max_y > PADDING else PADDING
        
        draw.text((x, y), text, fill=font_color, font=font)
        
        # 裁剪逻辑
        crop_x1 = max(0, x - PADDING)
        crop_y1 = max(0, y - PADDING)
        crop_x2 = min(CANVAS_SIZE[0], x + text_w + PADDING)
        crop_y2 = min(CANVAS_SIZE[1], y + text_h + PADDING)
        
        image_cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        image_cropped.save(output_path)

def main():
    print("🚀 启动全量字体生成脚本...")
    gen = FontGenerator()
    font_names = list(gen.fonts.keys())
    num_fonts = len(font_names)
    
    if num_fonts == 0: return

    # ✅ 修改：计算总量 = 字体数 * 每字体张数
    total_tasks = num_fonts * IMAGES_PER_FONT
    print(f"📊 任务计划: {num_fonts} 类字体 x {IMAGES_PER_FONT} 张/类 = 总计 {total_tasks} 张")
    
    pbar = tqdm(total=total_tasks)
    
    for f_name in font_names:
        save_dir = Path(OUTPUT_DIR) / f_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # ✅ 修改：固定循环 200 次
        for i in range(IMAGES_PER_FONT):
            save_path = save_dir / f"{i}.jpg"
            try:
                gen.generate(f_name, save_path)
                pbar.update(1)
            except Exception as e:
                # 打印错误但不中断，防止某个坏字体卡死整个流程
                # print(f"Error generating {f_name}: {e}") 
                pbar.update(1) 
                pass
                
    pbar.close()
    print(f"\n🎉 全部完成！数据集路径: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
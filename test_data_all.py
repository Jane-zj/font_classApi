"""
字体分类数据集生成器 (全量字体版)
功能：指定一组词语，使用文件夹内【所有】字体生成样本。
流程：
1. 扫描文件夹内所有字体 -> 遍历词语。
2. 绘制：在 800x800 大画布上绘制。
3. 裁剪：紧贴文字裁剪掉多余背景。
4. 输出：保存为 "{词语}_{序号}.jpg"。
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

# 1. 🔥🔥🔥 在这里填入你想生成的词语 🔥🔥🔥
TARGET_WORDS = ["接送", "特辣", "越一律师", "沈繁亮", "135-9217-8527", "ctshyh@163.com"]

# 2. 核心算法参数
CANVAS_SIZE = (800, 800)     # 临时画布
BACKGROUND_RATIO = 0.5       # 50% 使用图片背景
CONTRAST_COLOR_RATIO = 0.8   # 80% 保证高对比度
FONT_SIZE_RANGE = (50, 120)  # 字号范围
PADDING = 2                  # 裁剪时保留的边缘 (像素)

# 3. 路径配置
FONTS_DIR = "/root/autodl-tmp/font-classify/fonts"
BG_DIR = "/root/autodl-tmp/font-classify/sample_data/backgrounds/"
# 输出目录
OUTPUT_DIR = "/root/autodl-tmp/font-classify/test_all_fonts"

# =======================================================

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
        
        print(f"🔍 正在扫描字体目录: {FONTS_DIR}")
        loaded_count = 0
        
        # 🔥 修改点：遍历目录，加载所有支持的字体文件
        for root, _, files in os.walk(FONTS_DIR):
            for file in files:
                if file.lower().endswith((".ttf", ".otf", ".woff", ".woff2")):
                    name = os.path.splitext(file)[0]
                    # 直接加载，不再校验白名单
                    self.fonts[name] = os.path.join(root, file)
                    # 打印信息太长可以注释掉下面这行
                    # print(f"   ✅ 加载: {name}") 
                    loaded_count += 1
                    
        print(f"📦 共加载了 {loaded_count} 个字体文件")
        
        if loaded_count == 0:
            print("❌ 错误: 目录内未找到任何字体文件！")
            sys.exit(1)

    def get_font(self, name, size):
        key = f"{name}_{size}"
        if key in self.fonts_cache: return self.fonts_cache[key]
        try:
            font = ImageFont.truetype(self.fonts[name], size)
        except:
            # 备用方案
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

    def generate(self, font_name, output_path, text):
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
        
        # 绘制文本
        bbox = font.getbbox(text)
        text_w, text_h = bbox[2], bbox[3]
        
        # 简单的位置随机
        max_x = max(PADDING, CANVAS_SIZE[0] - text_w - PADDING)
        max_y = max(PADDING, CANVAS_SIZE[1] - text_h - PADDING)
        x = random.randint(PADDING, max_x) if max_x > PADDING else PADDING
        y = random.randint(PADDING, max_y) if max_y > PADDING else PADDING
        
        draw.text((x, y), text, fill=font_color, font=font)
        
        # 裁剪
        crop_x1 = max(0, x - PADDING)
        crop_y1 = max(0, y - PADDING)
        crop_x2 = min(CANVAS_SIZE[0], x + text_w + PADDING)
        crop_y2 = min(CANVAS_SIZE[1], y + text_h + PADDING)
        
        image_cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        image_cropped.save(output_path)

def main():
    print(f"🚀 启动全量字体生成脚本...")
    print(f"📝 目标词语: {TARGET_WORDS}")
    
    gen = FontGenerator()
    font_names = list(gen.fonts.keys())
    
    # 计算总数 = 字体数 x 词语数
    total_tasks = len(font_names) * len(TARGET_WORDS)
    print(f"📊 任务计划: {len(font_names)} 类字体 x {len(TARGET_WORDS)} 个词语 = {total_tasks} 张图片")
    
    pbar = tqdm(total=total_tasks)
    
    # 遍历每种字体
    for f_name in font_names:
        save_dir = Path(OUTPUT_DIR) / f_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 遍历每个目标词语
        for i, word in enumerate(TARGET_WORDS):
            
            # 🔥 注意：如果你文件夹里有纯英文字体（如 Arial），生成中文会变成方框
            # 如果你不想跳过任何字体，可以保留下面的 try-except 强行生成
            # 如果想跳过特定英文名字体生成中文，可以在这里加判断
            
            save_path = save_dir / f"{i}_{word}.jpg"
            
            try:
                gen.generate(f_name, save_path, text=word)
                pbar.update(1)
            except Exception as e:
                # print(f"Error on {f_name}: {e}") # 报错太多可以注释掉
                pbar.update(1)
                pass
                
    pbar.close()
    print(f"\n🎉 全部完成！去看看吧: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
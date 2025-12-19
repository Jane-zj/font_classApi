import random
import os

# === 1. 基础词库 ===
SURNAMES = list("赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐")
NAMES = ["建华", "文博", "志强", "国强", "明杰", "新伟", "家豪", "红梅", "秀英", "玉兰", "建国", "克林", "富贵", "伟", "敏", "静", "杰", "磊", "洋", "艳"]
TITLES = ["总经理", "CEO", "创始人", "设计总监", "高级工程师", "会计师", "业务经理", "销售代表", "行政主管", "合伙人", "董事长", "总监"]

CITIES = ["北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "武汉", "济南", "青岛", "苏州", "天津", "重庆", "西安"]
INDUSTRIES = ["科技", "网络", "文化传媒", "实业", "商贸", "广告", "咨询", "物流", "装饰", "教育", "餐饮管理", "电子", "生物技术"]
SUFFIXES = ["有限公司", "有限责任公司", "集团", "工作室", "中心", "事务所"]

ROADS = ["人民路", "建设路", "解放路", "和平路", "中山路", "高新大道", "科技园路", "中关村大街", "南京路", "世纪大道"]
BUILDINGS = ["财富中心", "国贸大厦", "SOHO", "科技大厦", "创意园", "金融中心", "广场"]

# === 2. 生成函数 ===

def gen_phone():
    """生成随机手机号"""
    prefix = random.choice(["138", "139", "136", "137", "150", "151", "186", "188", "189", "177"])
    return f"{prefix}{''.join(random.choices('0123456789', k=8))}"

def gen_tel():
    """生成座机号"""
    area = random.choice(["010", "021", "0755", "0571", "028", "0531"])
    return f"{area}-{''.join(random.choices('0123456789', k=8))}"

def gen_name_title():
    name = random.choice(SURNAMES) + "".join(random.choices(NAMES, k=1))
    return f"{name} {random.choice(TITLES)}"

def gen_company():
    return f"{random.choice(CITIES)}{random.choice(INDUSTRIES)}{random.choice(SUFFIXES)}"

def gen_address():
    no = random.randint(1, 999)
    room = random.randint(101, 2808)
    return f"{random.choice(CITIES)}市{random.choice(ROADS)}{no}号{random.choice(BUILDINGS)}{room}室"

def gen_email():
    """生成模拟邮箱"""
    user = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(5, 8)))
    domain = random.choice(["qq.com", "163.com", "gmail.com", "company.cn", "tech.com"])
    return f"{user}@{domain}"

# === 3. 生成并写入文件 ===
OUTPUT_FILE = "card_corpus.txt"
TOTAL_LINES = 5000  # 生成5000行

print(f"正在生成 {TOTAL_LINES} 条名片语料...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for _ in range(TOTAL_LINES):
        # 随机组合成一行文本
        mode = random.randint(1, 4)
        
        if mode == 1:
            # 模式1：全套信息
            line = f"{gen_name_title()} {gen_phone()} {gen_company()}"
        elif mode == 2:
            # 模式2：公司 + 地址 + 座机 (数字和汉字混合)
            line = f"{gen_company()} 地址:{gen_address()} 电话:{gen_tel()}"
        elif mode == 3:
            # 模式3：纯数字密集型 (对训练数字识别很有用)
            line = f"手机:{gen_phone()} 电话:{gen_tel()} 传真:{gen_tel()}"
        else:
            # 模式4：英文混合 (训练邮箱和网址)
            line = f"E-mail:{gen_email()} 网址:www.{random.choice(['baidu','sina','google'])}.com"
            
        f.write(line + "\n")

print(f"✅ 生成完成！文件位置: {os.path.abspath(OUTPUT_FILE)}")
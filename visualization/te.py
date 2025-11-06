# -*- codeing = utf-8 -*-
# ！E/reserve/python
# @Time :2025/11/6 12:28
# @Author:zcy
# @File : te.py
# @Software: PyCharm
try:
    import multipart
    print("✅ python-multipart 安装成功！")
    print(f"版本路径: {multipart.__file__}")
except ImportError as e:
    print("❌ python-multipart 安装失败")
    print(f"错误: {e}")
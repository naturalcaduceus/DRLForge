# 全局配置文件


import os
import sys
from pathlib import Path
# 把上级目录加入python搜索路径
sys.path.append(str(Path(__file__)))
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

# 将项目根目录添加到模块搜索路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import time
import logging
parser = argparse.ArgumentParser(description='Multi-task Network')
# pwdup = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))#当前文件的绝对路径的上一层
pwdup = os.path.abspath(os.path.dirname(__file__))#当前文件的绝对路径
parser.add_argument('--propath', default=pwdup+'/', type=str, help='项目根目录')
parser.add_argument('--propath2', default=pwdup+'/mywork/', type=str, help='项目根目录2')
parser.add_argument('--outputpath', default=pwdup+'/output/', type=str, help='杂项输出目录')

gopt = parser.parse_args()
# from config import gopt
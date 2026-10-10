import pandas as pd
import numpy as np
import matplotlib
import sklearn
import scipy
import platform

print("="*30)
print("竞赛论文环境版本检测报告")
print("="*30)
print(f"操作系统: {platform.system()} {platform.release()}")
print(f"Python 版本: {platform.python_version()}")
print("-"*30)
print(f"pandas: {pd.__version__}")
print(f"numpy: {np.__version__}")
print(f"matplotlib: {matplotlib.__version__}")
print(f"scikit-learn (sklearn): {sklearn.__version__}")
print(f"scipy: {scipy.__version__}")
print("="*30)
print("提示：请将以上信息完整复制到你的 README.txt 或备注文件中。")
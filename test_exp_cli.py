import os
os.environ["ADAPTIVE_AI_MODEL_PATH"] = r"C:\Orion\models\rwkv-7-world-0.4B-Q4_0.bin"
os.environ["RWKV_CPP_LIB"] = r"C:\Orion\rwkv.cpp\build\bin\librwkv.dll"

import sys

# Сначала выполним несколько задач
sys.argv = ["adaptive-ai", "solve", "Рассчитай 2+2"]
from adaptive_ai.app.main import main
print("=== solve 1 ===")
main()

sys.argv = ["adaptive-ai", "solve", "Привет"]
print("\n=== solve 2 ===")
main()

# Просмотр статистики опыта
sys.argv = ["adaptive-ai", "experience", "stats"]
print("\n=== experience stats ===")
main()

# Просмотр списка
sys.argv = ["adaptive-ai", "experience", "list"]
print("\n=== experience list ===")
main()

import os
import runpy

# 快速验证：覆盖超参数，减少训练时间与内存
os.environ["NUM_EPOCHS"] = os.environ.get("NUM_EPOCHS", "1")
os.environ["BATCH_SIZE"] = os.environ.get("BATCH_SIZE", "32")

print("Running quick verification: NUM_EPOCHS=", os.environ["NUM_EPOCHS"], "BATCH_SIZE=", os.environ["BATCH_SIZE"])

# 以脚本方式执行 train_and_eval.py（会读取上面设置的环境变量）
runpy.run_path("train_and_eval.py", run_name="__main__")

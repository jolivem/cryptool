# check_tf_gpu.py
import os
# Optional: show TF device logs (must be before importing TF)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

import tensorflow as tf

print("TF version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Physical GPUs:", tf.config.list_physical_devices("GPU"))
print("Logical GPUs:", tf.config.list_logical_devices("GPU"))
print("Build info:", {k: v for k, v in tf.sysconfig.get_build_info().items() if "cuda" in k.lower() or "cudnn" in k.lower()})

# Log device placement for ops
tf.debugging.set_log_device_placement(True)

# Tiny compute to force device use
with tf.device("/GPU:0"):
    a = tf.random.normal([8000, 8000])
    b = tf.random.normal([8000, 8000])
    c = tf.matmul(a, b)      # you should see logs indicating GPU:0
    _ = c.numpy()            # materialize the result

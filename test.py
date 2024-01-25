import tensorflow as tf
import torch
import onnx
import onnxruntime 

def check_tensorflow_gpu():
    if tf.test.is_gpu_available():
        print("TensorFlow GPU support is available.")
        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            print(f"GPU Name: {device.name}")
    else:
        print("TensorFlow GPU not found. Using CPU.")

def check_pytorch_gpu():
    if torch.cuda.is_available():
        print("PyTorch GPU support is available.")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU Name: {gpu_name}")
    else:
        print("PyTorch GPU not found. Using CPU.")

def check_onnx_gpu():
    device = onnxruntime.get_device()
    print(device)
    if device == 'GPU':
        print("ONNX GPU support is available.")
        gpu_name = onnxruntime.get_available_providers()
        print(f"Available Execution Providers: {gpu_name}")
    else:
        print("ONNX GPU not found. Using CPU.")

if __name__ == "__main__":
    print("Checking TensorFlow GPU support:")
    check_tensorflow_gpu()

    print("\nChecking PyTorch GPU support:")
    check_pytorch_gpu()

    print("\nChecking ONNX GPU support:")
    check_onnx_gpu()

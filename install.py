import launch

if not launch.is_installed("diffusers"):
    launch.run_pip(f"install diffusers", "diffusers")
if not launch.is_installed("onnxruntime-gpu"):
    launch.run_pip(f"install onnxruntime-gpu", "onnxruntime-gpu")

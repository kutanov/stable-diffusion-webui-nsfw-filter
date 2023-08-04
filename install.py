import launch

if not launch.is_installed("diffusers"):
    launch.run_pip(f"install diffusers", "diffusers")
if not launch.is_installed("onnxruntime-gpu"):
    launch.run_pip(f"install onnxruntime-gpu", "onnxruntime-gpu")
if not launch.is_installed("open_clip_torch"):
    launch.run_pip(f"install open_clip_torch", "open_clip_torch")

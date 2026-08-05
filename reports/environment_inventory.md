# M0 environment inventory

Generated: `2026-08-05T05:34:22.325642+00:00`

This report records non-destructive observations only. No process was terminated and no system environment was modified.

## Host summary

| Host | SSH/probe | OS | CPU | GPU summary | PyTorch | Slurm |
|---|---|---|---:|---|---|---|
| doraemon02 | SUCCEEDED | Linux-5.15.0-130-generic-x86_64-with-glibc2.35 | 16 | 0, GPU-385d2fc4-943f-a9fd-640d-a949bf24421b, Quadro P6000, 24576, 24435, 0, 565.57.01<br>1, GPU-b1764cf2-a58c-5d4b-f5c8-acec9e69d507, Quadro P6000, 24576, 24435, 0, 565.57.01<br>2, GPU-a0b88c77-ad10-5431-01dd-5aaba1b7575c, Quadro P6000, 24576, 24435, 0, 565.57.01<br>3, GPU-3256e567-e3ac-08a6-caf2-9eae89dfac34, Quadro P6000, 24576, 24435, 0, 565.57.01 | unavailable: Traceback (most recent call last):<br>  File "<string>", line 1, in <module><br>ModuleNotFoundError: No module named 'torch' | no |
| doraemon03 | SUCCEEDED | Linux-5.15.0-130-generic-x86_64-with-glibc2.35 | 16 | 0, GPU-b56daa58-4567-25a2-4b39-2031ae6bd5cb, NVIDIA GeForce RTX 3090, 24576, 24148, 0, 565.57.01<br>1, GPU-a329ebc3-1898-5261-f967-9a3d50c53d9a, NVIDIA GeForce RTX 3090, 24576, 24150, 0, 565.57.01 | unavailable: Traceback (most recent call last):<br>  File "<string>", line 1, in <module><br>ModuleNotFoundError: No module named 'torch' | no |
| doraemon04 | SUCCEEDED | Linux-5.15.0-130-generic-x86_64-with-glibc2.35 | 16 | 0, GPU-befba2d1-d08d-a842-ed49-e05cba9e0e74, NVIDIA GeForce GTX 1080 Ti, 11264, 11170, 0, 535.183.01<br>1, GPU-3c2e0e76-d537-ad18-43e5-820ffa438644, NVIDIA GeForce GTX 1080 Ti, 11264, 11170, 0, 535.183.01<br>2, GPU-8eb71e2d-5528-5387-a211-0eb2b0ce591a, NVIDIA GeForce GTX 1080 Ti, 11264, 11170, 0, 535.183.01<br>3, GPU-9721fdfd-81c6-dda3-da6d-e55bed90ab26, NVIDIA GeForce GTX 1080 Ti, 11264, 11170, 0, 535.183.01 | unavailable: Traceback (most recent call last):<br>  File "<string>", line 1, in <module><br>ModuleNotFoundError: No module named 'torch' | no |
| doraemon15 | SUCCEEDED | Linux-5.4.0-90-generic-x86_64-with-glibc2.29 | 80 | 0, GPU-180748c0-8783-fe7a-300f-5dfc34f5cb06, Quadro RTX 8000, 48601, 48600, 0, 495.29.05<br>1, GPU-23e05330-b168-8956-7206-0aa8f9aaa82b, Quadro RTX 8000, 48601, 48600, 0, 495.29.05<br>2, GPU-81e10404-c033-98ed-4aea-cf544e3a59cf, Quadro RTX 8000, 48601, 48600, 0, 495.29.05<br>3, GPU-7e109458-ab21-aba5-8721-f7c90e172e6c, Quadro RTX 8000, 48601, 48600, 0, 495.29.05 | 1.10.0+cu102 | no |
| doraemon19 | TIMEOUT: SSH probe failed without output | — | — | — | — | — |
| doraemon20 | SUCCEEDED | Linux-6.8.0-88-generic-x86_64-with-glibc2.39 | 96 | 0, GPU-9b09df6b-3a73-f9da-36be-372738472356, NVIDIA A100 80GB PCIe, 81920, 29396, 0, 580.95.05<br>1, GPU-60511ca1-d7a9-2b7d-6108-069aee86d5e5, NVIDIA A100 80GB PCIe, 81920, 20776, 0, 580.95.05<br>2, GPU-a4da8020-056b-8d97-9171-9ac5f8fcbd30, NVIDIA A100 80GB PCIe, 81920, 15668, 100, 580.95.05<br>3, GPU-588bfda8-5c95-35ea-9c0b-84c5fc1963cf, NVIDIA A100 80GB PCIe, 81920, 79550, 17, 580.95.05<br>4, GPU-30088f79-55bb-af3d-f64a-b3c304095a05, NVIDIA A100 80GB PCIe, 81920, 81148, 0, 580.95.05<br>5, GPU-3b579700-80d9-08ce-03c2-85d64d6db432, NVIDIA A100 80GB PCIe, 81920, 75868, 0, 580.95.05<br>6, GPU-f17c0166-5452-4395-8622-b4b6911410c5, NVIDIA A100 80GB PCIe, 81920, 43570, 22, 580.95.05<br>7, GPU-658e397f-0384-4379-4448-7ced780e215b, NVIDIA A100 80GB PCIe, 81920, 76610, 0, 580.95.05 | unavailable: Traceback (most recent call last):<br>  File "<string>", line 1, in <module><br>ModuleNotFoundError: No module named 'torch' | no |

## Windows development host

- Host: `lzhang`
- OS: `Windows-11-10.0.26200-SP0`
- Logical CPUs: `16`
- Python: `3.13.12 | packaged by conda-forge | (main, Feb  5 2026, 05:41:12) [MSC v.1944 64 bit (AMD64)]`
- PyTorch: `{'available': True, 'version': '2.13.0+cpu', 'cuda_version': None, 'cuda_available': False, 'gpu_count': 0}`
- GPU: `0, GPU-f8c4e02b-78e7-6487-21fa-6743039755fc, NVIDIA GeForce RTX 5070 Ti, 16303, 9669, 21, 610.47`

## Shared path decision

- Status: **UNRESOLVED**
- Reason: No identical configured project root was observed on all six hosts.
- Common mounts on reachable hosts: `{'/homes': ['taka2:/homes'], '/suedata1': ['taka2:/suedata1'], '/data4': ['taka2:/data4'], '/data7': ['taka9:/data7'], '/dataT0': ['taka2new:/dataT0'], '/dataT1': ['taka2new:/dataT1'], '/dataF0': ['taka2new:/dataF0'], '/suedata1copy': ['taka2new:/fcbackup06'], '/homescopy': ['taka2new:/fcbackup08'], '/data4copy': ['taka2new:/fcbackup09']}`
- `configs/hosts.env` remains local and its root paths remain empty until all six hosts can be checked.

## Scheduler and first-run decision

- Slurm was not detected on successfully probed hosts; a conservative SSH dispatcher is the applicable future scheduler.
- Provisional least-disruptive host by idle-GPU count, then free memory: `doraemon15`.
- No Linux training is authorized until `TAMOE_PROJECT_ROOT`, data, run, and cache roots are confirmed across all six hosts.

Full command outputs, process lists, memory, driver/CUDA, disk, mount, and path observations are preserved in `reports/environment_inventory.json`.

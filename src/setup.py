from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
# 把 quadsim.cpp（C++ 绑定文件）和两个 .cu 文件一起编译
# 变成一个可以被 Python 导入的模块 quadsim_cuda
setup(
    name='quadsim_cuda',
    ext_modules=[
        CUDAExtension('quadsim_cuda', [
            'quadsim.cpp',  # C++ 包装/绑定文件（通常含 PYBIND11_MODULE）
            'quadsim_kernel.cu',    # CUDA 实现文件 1
            'dynamics_kernel.cu',   # CUDA 实现文件 1
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension # 使用 PyTorch 提供的编译器配置
    })

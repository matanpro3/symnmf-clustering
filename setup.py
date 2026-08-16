from setuptools import setup, Extension

# Define the C extension module
symnmf_module = Extension(
    'symnmf',
    sources=['symnmfmodule.c', 'symnmf.c'],
    # You might need to add other arguments for specific compilers or platforms
    # For example, to specify a standard: extra_compile_args=['-std=c99']
)

setup(
    name='symnmf',
    version='1.0',
    author='Your Name',
    description='A Python package for Symmetric Non-negative Matrix Factorization (symNMF)',
    ext_modules=[symnmf_module]
)

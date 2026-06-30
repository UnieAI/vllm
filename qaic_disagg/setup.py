from setuptools import setup, find_packages

setup(
    name="qaic_disagg",
    version="0.0.1",
    description="Qaic Cache: manages prefill storage and communication",
    author="Qualcomm Cloud AI Team",
    packages=find_packages(),
    install_requires=[
        "vllm",
        "numpy",
        "pyzmq",
        "msgspec"
        "rich"
        # List any dependencies here
        # e.g., "numpy", "requests"
    ],
    python_requires=">=3.10",
    entry_points={
        'console_scripts': [
            
            # Add command-line scripts here
            # e.g., "my_command=my_package.module:function"
        ],
    },
)


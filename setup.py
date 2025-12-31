from setuptools import setup, find_packages

setup(
    name="scvalue",
    version="0.1.0",
    author="Li Huang",
    author_email="hl@ism.cams.cn",
    description="scValue: value-based subsampling of large-scale single-cell transcriptomic data for machine and deep learning tasks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/lhbcb/scvalue",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=1.5.2,<2.0",
        "scipy>=1.14.0,<2.0",
        "numpy>=1.26.4,<3.0",
        "pandas>=1.5.3,<3.0",
        "joblib>=1.4.2,<2.0",
        "scanpy>=1.10.2,<2.0",
        "anndata>=0.10.8,<0.13",
    ],
    license='BSD-3-Clause',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)


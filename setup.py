import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="FredMD",
    version="0.0.4",
    author="Joe Saia",
    author_email="joe5saia@gmail.com",
    description="Estimate factors off of the FRED-MD dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/joe5saia/FredMD",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['pandas>=1.2.0', 'numpy>=1.19.0', 'scikit-learn>=0.24.0'],
    python_requires='>=3.6',
)

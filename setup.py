from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="chains-py",
    version="0.2.0",
    author="ohadr",
    description="A unified, chainable interface for working with multiple LLM providers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ohadr/chains",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "anthropic",
        "anthropic[bedrock]",
        "google-generativeai>=0.3.0",
        "openai",
        "tenacity",
        "appdirs",
        "pydantic",
        "fire",
        "instructor",
        "jinja2",
    ],
)

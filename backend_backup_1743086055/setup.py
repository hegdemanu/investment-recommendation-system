from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="investment-recommendation-system",
    version="1.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive system for stock price prediction and investment recommendations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/investment-recommendation-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "investment-system=run:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "app": ["templates/*", "static/*"],
    },
) 
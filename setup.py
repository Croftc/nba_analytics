from setuptools import setup, find_packages

setup(
    name="nba_modeling",  # Replace with your package name
    version="0.1",
    packages=find_packages(),
    install_requires=[],  # List any dependencies here
    include_package_data=True,
    description="A project with utilities for backtesting and analysis",
    author="Goku Moneymaker",
    author_email="your_email@example.com",
    #url="https://github.com/yourusername/my_project",  # Optional
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)

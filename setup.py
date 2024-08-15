from setuptools import find_packages, setup

setup(
    name="hybrid_ode_sim",
    version="0.1",
    packages=find_packages(),
    install_requires=[], # Install requirements with pip install -r requirements.txt
    author="Micah Reich",
    author_email="micahreich02@gmail.com",
    description="Hybrid continuous/discrete ordinary differential equation solver written in Python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/micahreich/hybrid_ode_sim",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9.6",
)

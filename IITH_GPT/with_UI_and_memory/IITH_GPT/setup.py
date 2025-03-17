from setuptools import setup, find_packages

# Print the packages that find_packages() detects
print("Detected packages:", find_packages())

setup(
    name="IITH_GPT",  # Name of the package
    version="0.1",  # Package version
    packages=find_packages(),  # Automatically find all packages
    install_requires=[  # List of dependencies (if any)
        # "some_package",  # Example
    ],
)

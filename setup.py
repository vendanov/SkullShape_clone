from setuptools import setup, find_packages
import os

# Read the contents of README.md for the long description
def read_long_description():
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        return f.read()
# Collect all Python packages excluding certain directories
def find_project_packages():
    return find_packages(
        where='.',
        exclude=[
            'tests',
            'notebooks',
            'runner_scripts',
            'smk_configs',
            'test_notebooks',
            'matlab',
            'models',
            'support',
            'external_libraries',
            'venv',
            'env',
        ]
    )

setup(
    name='mouse_skull_pipelines',  # Replace with your desired package name
    version='0.1.0',  # Initial version
    author='Your Name',  # Replace with your name
    author_email='your.email@example.com',  # Replace with your email
    description='A pipeline for analyzing mouse skull data using Snakemake workflows and machine learning models.',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/mouseSkullPipelines',  # Replace with your project's URL
    packages=find_packages(include=['mouse_skull_pipelines', 'mouse_skull_pipelines.*']),
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'snakemake',
        'typer',
        # Add other dependencies as needed
    ],
    extras_require={
        'dev': [
            'pytest',
            'flake8',
            'black',
            # Add other development dependencies
        ],
        'docs': [
            'sphinx',
            'mkdocs',
            # Add other documentation dependencies
        ],
    },
    entry_points={
        'console_scripts': [
            'run-mouse-skull=main:main',  # Ensure main.py has a main() function
            'convert-files=convert_files:main',  # If convert_files.py has a main() function
            'clean-mouse-skull=clean_mouse_skull:main',  # Similarly for other scripts
            'run-dxa-fantoms=run_dxa_fantoms:main',
            'run-mouse-sutures-train=run_mouse_sutures_train:main',
            'run-mouseskull=run_mouseskull:main',
            'run-tmd-bmd=run_tmd_bmd:main',
            # Add other scripts as needed
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3.11',  # Specify your Python versions
        'License :: OSI Approved :: MIT License',  # Replace with your license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',  # Specify the minimum Python version
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/mouseSkullPipelines/issues',
        'Source': 'https://github.com/yourusername/mouseSkullPipelines',
    },
)

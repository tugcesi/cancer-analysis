from setuptools import setup, find_packages

setup(
    name="cancer-prognosis",
    version="1.0.0",
    author="Tugcesi",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'streamlit',
        'plotly',
        'joblib',
        'jupyter',
    ],
)
from setuptools import setup, find_packages

setup(
    name='ozone_level_detection',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas==1.3.3',
        'google-cloud-storage==1.42.3',
        'google-cloud-aiplatform==1.7.1',
        'tensorflow==2.15',  # Specifying TensorFlow 2.15
        'numpy==1.21.2',
        'pickle5==0.0.11',
        'protobuf==3.19.6',
        'python-json-logger==2.0.4',  # Include python-json-logger
    ],
)

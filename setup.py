from setuptools import setup, find_packages

setup(
  name = 'classification_models_pytorch',
  packages = find_packages(exclude=['examples']),
  version = '0.0.1',
  license='MIT',
  description = 'Classification_models_pytorch',
  long_description_content_type = 'text/markdown',
  author = 'Seungmin Ha & Saerom Park',
  author_email = 'smha@promedius.ai',
  url = 'https://github.com/smha-Promedius/classification_models_pytorch',
  keywords = [
    'artificial intelligence',
    'image classification',
    'torchvision'
  ],
  install_requires=[
    'torch>=1.10',
    'torchvision'
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  classifiers=[
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.8',
  ],
)
from setuptools import find_packages, setup

setup(
    name = 'transformers_sklearn',
    version = "1.0.0",
    author = 'trueto',
    author_email='3130103836@zju.edu.cn',
    description="A sklearn wrapper for Transformers",
    keywords='scikit_sklearn Transformers NLP Deep Learning',
    license='Apache',
    url='https://github.com/trueto/transformers_sklearn',
    packages=find_packages(exclude=['examples']),
    install_requires=['torch>=1.0.0',
                      'transformers>=2.2.0',
                      'scikit-learn',
                      'numpy',
                      'pandas',
                      'boto3',
                      'requests',
                      'tqdm',
                      'tensorboardx'
                      ],
    python_requires='>=3.5.0',
    classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
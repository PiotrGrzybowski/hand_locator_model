from setuptools import setup, find_packages

setup(name='hand_locator_model',
      version='1.0.1',
      packages=find_packages(),
      install_requires=['numpy>=1.13.3',
                        'tensorflow-gpu==1.4.0',
                        'wget>=3.2'],
      author='Piotr Grzybowski',
      author_email='p.grzybowski2@gmail.com',
      description='Hand tracking model for Sign Language Recognition System',
      license='MIT',
      keywords='hand detection',
      url='https://'
      )

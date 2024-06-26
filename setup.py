from setuptools import setup, find_packages

setup(
    name='EVbatteryDetection',
    version='0.1.0',
    author='Claudio Cimarelli',
    description='A vision system for battery disassembly automation.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/claudiocimarelli/EVbatteryDetection',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=open('requirements.txt').read().splitlines(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
    entry_points={
        'console_scripts': [
            'battery-vision=main:main',
        ],
    },
)

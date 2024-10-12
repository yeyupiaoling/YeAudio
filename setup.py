from setuptools import setup, find_packages

import yeaudio

VERSION = yeaudio.__version__


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def parse_requirements():
    with open('./requirements.txt', encoding="utf-8") as f:
        requirements = f.readlines()
    return requirements


if __name__ == "__main__":
    setup(
        name='yeaudio',
        packages=find_packages(),
        package_data={'': ['utils/vad_models/*']},
        author='yeyupiaoling',
        version=VERSION,
        install_requires=parse_requirements(),
        description='Audio ToolKit for Python',
        long_description=readme(),
        long_description_content_type='text/markdown',
        url='https://github.com/yeyupiaoling/YeAudio',
        download_url='https://github.com/yeyupiaoling/YeAudio.git',
        keywords=['audio', 'speech'],
        classifiers=[
            'Intended Audience :: Developers',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Natural Language :: Chinese (Simplified)',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
            'Topic :: Utilities'
        ],
        license='Apache License 2.0',
        ext_modules=[])

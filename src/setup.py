import io
import re
from setuptools import setup, find_packages

from kalman import __version__

def read(file_path):
    with io.open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


readme = read('README.rst')
# вычищаем локальные версии из файла requirements (согласно PEP440)
requirements = '\n'.join(
    re.findall(r'^([^\s^+]+).*$',
               read('requirements.txt'),
               flags=re.MULTILINE))


setup(
    # metadata
    name='kalman',
    version=__version__,
    license='MIT',
    author='Matvei Kreinin, Maria Nikitina, Petr Babkin, Anastasia Voznyuk',
    author_email="kreinin.mv@phystech.edu, nikitina.mariia@phystech.edu, babkin.pk@phystech.eduб vozniuk.ae@phystech.edu",
    description='Kalman filter and his friends',
    long_description=readme,
    url='https://github.com/intsystems/Kalman-filter-and-his-friends',

    # options
    packages=find_packages(),
    install_requires=requirements,
)


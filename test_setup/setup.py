# coding:utf-8
from setuptools import setup
setup(name='app',
      version='0.1',
      packages=['app'])


'''
ref:
http://www.bjhee.com/setuptools.html

python setup.py install:
该命令会将当前的Python应用安装到当前Python环境的”site-packages”目录下，这样其他程序就可以像导入标准库一样导入该应用的代码了。
python setup.py develop:
如果应用在开发过程中会频繁变更，每次安装还需要先将原来的版本卸掉，很麻烦。使用”develop”开发方式安装的话，应用代码不会真的被拷贝到本地Python环境的”site-packages”目录下，而是在”site-packages”目录里创建一个指向当前应用位置的链接。这样如果当前位置的源码被改动，就会马上反映到”site-packages”里。
'''

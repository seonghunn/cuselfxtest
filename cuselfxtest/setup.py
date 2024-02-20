from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
import os
import subprocess

class CustomBuild:
    def execute_build(self):
        # 프로젝트의 외부 의존성과 빌드 과정을 설정
        if os.path.isdir("external"):
            print("Removing existing 'external' directory...")
            subprocess.check_call(['rm', '-rf', 'external'])
        
        if os.path.isdir("build"):
            print("Removing existing 'build' directory...")
            subprocess.check_call(['rm', '-rf', 'build'])

        print("Cloning libigl into ./external...")
        subprocess.check_call(['git', 'clone', 'https://github.com/libigl/libigl.git', 'external/libigl'])

        os.environ['LIBIGL_DIR'] = os.path.join(os.getcwd(), 'external/libigl')
        os.environ['PATH'] += os.pathsep + '/usr/local/cuda/bin'
        os.environ['CUDACXX'] = '/usr/local/cuda/bin/nvcc'
        os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64' + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['CUDA_PATH'] = '/usr/local/cuda'
        os.environ['CUDA_HOME'] = '/usr/local/cuda'

        if not os.path.isdir("build"):
            os.makedirs("build")
        os.chdir("build")
        subprocess.check_call(['cmake', '..', '-DCMAKE_BUILD_TYPE=Release'])
        subprocess.check_call(['make'])
        os.chdir("..")  # Ensure to go back to the project root directory

class CustomInstall(CustomBuild, install):
    def run(self):
        self.execute_build()
        install.run(self)

class CustomDevelop(CustomBuild, develop):
    def run(self):
        self.execute_build()
        develop.run(self)

setup(
    name='cuselfxtest',
    version='1.0',
    packages=find_packages(),
    cmdclass={
        'install': CustomInstall,
        'develop': CustomDevelop,
    },
    install_requires=[
        # 여기에 필요한 Python 패키지 의존성 추가
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A C++ project with Python bindings for self-intersection tests using CUDA and libigl.',
    long_description='This is a package for cuselfxtest, a project that performs self-intersection tests on 3D models using CUDA and libigl.',
    url='https://github.com/yourgithubrepo/cuselfxtest',
)

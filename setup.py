# Copyright 2025 chen1110. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

install_requires = [
    "psutil>=5.8.0",
    "pynvml>=11.0.0",
    "requests>=2.25.0",
    "torch>=1.10.0",
]

extra_require = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.950",
    ],
    "dlrover": [
        "dlrover>=0.6.0",
    ],
}

setup(
    name="chen1110-rl-fault-tolerance",
    version="0.1.0",
    description="RL Training Fault Tolerance System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="chen1110",
    url="https://github.com/chen1110/rl-fault-tolerance",
    install_requires=install_requires,
    extras_require=extra_require,
    python_requires=">=3.8",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)


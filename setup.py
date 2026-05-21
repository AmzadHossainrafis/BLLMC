import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


def get_requirements():
    with open("requirements.txt", "r") as f:
        return f.read().splitlines()


__version__ = "0.0.0"

REPO_NAME = "BLLMC"
AUTHOR_USER_NAME = "Amzad hossain rafi"
SRC_REPO = "BLLMC"
AUTHOR_EMAIL = "[EMAIL_ADDRESS]"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for building Bangla LLM",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)

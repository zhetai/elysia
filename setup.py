from setuptools import find_packages, setup


def read_reqs(fname):
    all_requirements = []
    with open(fname) as req:
        all_requirements.extend(
            [
                line.strip()
                for line in req
                if line.strip()
                and not line.startswith("#")
                and not line.startswith("https")
            ]
        )
    return all_requirements


setup(
    name="elysia",
    version="0.1.0",
    python_requires=">=3.10.0,<3.13.0",
    author="Weaviate",
    author_email="danny@weaviate.io",
    description=(
        "Elysia is an open-source agentic platform for searching data. "
        "It is built with customisation in mind, allowing you to build agents "
        "and tools that are tailored to your specific use case. "
        "It uses Weaviate as the default retrieval tools, and can interface "
        "with your data stored in a Weaviate cluster."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=read_reqs("requirements.txt"),
    include_package_data=True,
    extras_require={
        "dev": read_reqs("requirements-dev.txt"),
    },
    entry_points={
        "console_scripts": [
            "elysia=elysia.api.cli:cli",
        ],
    },
)

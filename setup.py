from setuptools import setup, find_packages

setup(
    name="dbchat",
    version="0.1.0",
    description="",
    author="Sam Schumacher, Parmann Alizadeh",
    author_email="sam@onewaveisallittakes.com, prmma23@gmail.com",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
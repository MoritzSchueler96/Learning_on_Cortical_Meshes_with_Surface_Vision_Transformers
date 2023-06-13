from setuptools import setup

setup(
    name="cv_utils",
    version="0.0.1",
    url="https://github.com/MoritzSchueler96/TUM_AlzheimerPrediction",
    author="Moritz Schüler",
    author_email="moritz.schueler@tum.de",
    description="helper lib",
    packages=["cv_utils"],
    package_data={},
    license="MIT",
    long_description_content_type="text/markdown",
    long_description=""" helper lib """,
    install_requires=["pandas>=1.2.4"],
)

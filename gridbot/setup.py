from setuptools import find_packages, setup
from glob import glob
import os

package_name = "gridbot"


# helper to allow for easy import of models (as they come in their own folders)
def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths


models = []
for file in package_files("sim_assets"):
    dest = os.path.join("share", package_name, os.path.dirname(file))
    models.append((dest, [file]))


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.py*")),
        ),
        (
            os.path.join("share", package_name, "description"),
            glob(os.path.join("description", "*")),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob(os.path.join("config", "*")),
        ),
        (
            os.path.join("share", package_name, "sim_assets/worlds"),
            glob(os.path.join("sim_assets/worlds", "*")),
        ),
    ]
    + models,
    install_requires=["setuptools", "gpiozero", "opencv-python"],
    zip_safe=True,
    maintainer="Francis",
    maintainer_email="francis.grove@pm.me",
    description="TODO: Package description",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "grid_processor = gridbot.grid_processor:main",
            "route_navigator = gridbot.route_navigator:main",
            "motor_driver = gridbot.motor_driver:main",
            "map_generator = gridbot.map_generator:main",
        ],
    },
)

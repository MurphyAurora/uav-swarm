from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'xtd2_mission'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    package_data={
        'xtd2_mission': ['gazebo_assets/*.sdf'],
    },
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Andy Zhuo',
    maintainer_email='zhuoan@stu.pku.edu.cn',
    description='XTDrone2 mission and swarm algorithms',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'swarm_state_exchange = xtd2_mission.swarm_state_exchange:main',
            'uav_local_controller = xtd2_mission.uav_local_controller:main',
            'local_avoid_orca = xtd2_mission.local_avoid_orca:main',
            'dynamic_obstacle_source = xtd2_mission.dynamic_obstacle_source:main',
            'multi_waypoint2 = xtd2_mission.multi_waypoint2:main',
            'mission_controller = xtd2_mission.mission_controller:main',
            'obstacle_config = xtd2_mission.obstacle_config:main',
        ],
    },
)

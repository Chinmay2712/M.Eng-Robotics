from setuptools import find_packages, setup

package_name = 'cvaity_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pi',
    maintainer_email='chinmay.vaity09@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
				'image_subscriber = cvaity_vision.image_subscriber:main',
				'lane_center_node = cvaity_vision.lane_center_node:main',
            			'rect_direction_node = cvaity_vision.rect_direction_node:main',
            			'image_file_publisher = cvaity_vision.image_file_publisher:main',
            			'lane_center_node_flipped = cvaity_vision.lane_center_node_flipped:main',
            			'rect_direction_node_flipped = cvaity_vision.rect_direction_node_flipped:main',
				'lane_center_node_straight = cvaity_vision.lane_center_node_straight:main',
            			'rect_direction_node_straight = cvaity_vision.rect_direction_node_straight:main',
        ],
    },
)

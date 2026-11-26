from setuptools import find_packages, setup

package_name = 'cvaity_camera'

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
		'camera_publisher = cvaity_camera.camera_publisher:main',
            	'camera_publisher_flipped = cvaity_camera.camera_publisher_flipped:main',
            	'camera_publisher_straight = cvaity_camera.camera_publisher_straight:main',
        ],
    },
)

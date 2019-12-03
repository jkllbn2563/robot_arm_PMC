from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
from Cython.Build import cythonize
setup(
    #ext_modules=cythonize('/home/jkllbn2563/catkin_ws/src/robot_arm_PMC/src/cython/server.pyx')
    #ext_modules=cythonize('/home/jkllbn2563/catkin_ws/src/robot_arm_PMC/src/cython/client.pyx')
    ext_modules=cythonize('/home/jkllbn2563/catkin_ws/src/robot_arm_PMC/src/cython/total.pyx')
    )

setup_args = generate_distutils_setup(
    packages=['robot_arm_PMC'],
    package_dir={'':'src'},
    requires=['std_msgs', 'rospy','sensor_msgs']
)

setup(**setup_args) 
#python setup.py build_ext --inplace



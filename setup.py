from distutils.core import setup


setup(name = "pyclust",
    version = "0.1.0",
    description = "Data Clustering Algorithms in Python",
    author = "Vahid Mirjalili",
    author_email = "vmirjalily@gmail.com",
    url = "https://github.com/mirjalil/pyclust",

    packages = ['pyclust'],

    #install_requires=['numpy>=1.9.2',
    #	'scipy>=0.15.1',
    #	'treelib>=1.3.1'],

    #package *needs* these files.
    package_data = {'pyclust':[]},
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
	"Intended Audience :: Information Technology",
	"Operating System :: OS Independent",
    ],

    scripts = [],
    long_description = """


 Contact
=============

email: vmirjalily@gmail.com
Twitter: https://twitter.com/vmirly

URL for this project: https://github.com/mirjalil/pyclust

""", 
    #classifiers = [],
    license='GPLv3',
    platforms='any',
) 



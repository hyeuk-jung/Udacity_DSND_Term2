# Udacity_DSND_Term2
Udacity Data Scientist Nanodegree Projects (Term 2)

## Part 7

## Part 8-1: Object-Oriented Programming
Topics covered in this lesson are:
* classes and objects
* attributes and methods
* magic methods
* inheritance

Classes, object, attributes, methods, and inheritance are common to all object-oriented programming languages.

Here is a list of resources for advanced Python object-oriented programming topics.

* class methods, instance methods, and static methods - these are different types of methods that can be accessed at the class or object level
  (https://realpython.com/instance-class-and-static-methods-demystified/)

* class attributes vs instance attributes - you can also define attributes at the class level or at the instance level
  (https://www.python-course.eu/python3_class_and_instance_attributes.php)

* multiple inheritance, mixins - A class can inherit from multiple parent classes
  (https://easyaspython.com/mixins-for-fun-and-profit-cb9962760556)

* Python decorators - Decorators are a short-hand way for using functions inside other functions
  (https://realpython.com/primer-on-python-decorators/)


## Part 8-2: Modularizing
Topics covered in this lesson are:
* Modularizing code
* Making a package with Pip, a python package manager: 
 - Package: a collection of python modules
 - "__init__.py": Required file for a python package
 - "from .FileName import ClassName": Grammar for import statement in python3 (for .py files in a package)
 - "setup.py": Required file to make the folder in the same directory to be recognized as a package by pip
 - "PackageName.__file__": After you imported the package, this command will return the address where this package is installed 
 - "pip install .": A command line when you are trying to install the package (in the package directory, where setup.py exists)
 - "pip install --upgrade .": A command line to upgrade the pre-installed package (you should be in the package directory you want to update)
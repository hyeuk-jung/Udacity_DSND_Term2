# Udacity_DSND_Term2
Udacity Data Scientist Nanodegree Projects (Term 2)

# Part 7

# Part 8. Software Engineering
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

## Part 8-3: Web Development
 ### 1. Basics of a web app
   - HTML: Controls content 
     1. [Full List of Tags](https://www.w3schools.com/tags/default.asp)
     
     2. [W3C Validator](https://validator.w3.org/#validate_by_input)
     
     3. Div, Span, IDs, and Classes
   
   - CSS: Controls style
     1. [W3 Schools CSS Website](https://www.w3schools.com/css/default.asp)
     
     2. Examples: In the **head** tab, 
        ```html
        <head>
            <style>
                # id_name {
                    font-size: 20px; (or "font-size: 100%" for dynamic font size w.r.t. default browser font size)
                    font-weight: bold;
                    color: red;
                    text-decoration: none;
                    border: solid red 1px;
                    margin: 40px;                        
                    padding: 40px;
                }
                tab_name (eg. body / img / a) {styles you want to apply}
                .class_name {styles}
            </style>
        </head>
        ```
        ```html
        <head>
            <link rel="stylesheet" type"text/css" href="path to the .css file">
        </head>
        ```
   
   - JavaScript: Interactivity
     1. JavaScript: a high level language like Python, PHP, Ruby, and C++. It was specifically developed to make the front-end of a web application more dynamic.
        * Example: 
            ```html
            function addValues(x) { 
                var sum_array = 0; 
                for (var i=0; i < x.length; i++) {   
                    sum_array += x[i]; 
                } 
                return sum_array; 
            }

            addValues([3,4,5,6]);
            ```
            ```html
            function headFunction() { 
                document.getElementsByTagName("h1")[0].innerHTML = "A Photo of a Breathtaking View"; 
            }

            <img src="image.jpg" onclick="headFunction()">
            ```
     2. Basic syntax
        * a line of code ends with a semi-colon ;
        * () parenthesis are used when calling a function much like in Python
        * {} curly braces surround large chunks of code or are used when initializing dictionaries
        * [] square brackets are used for accessing values from arrays or dictionaries much like in Python
     
     3. jQuery: a JavaScript library that makes developing the front-end easier
        * Newer JavaScript tools: React and Angular
        * Example: 
            ```html
            $(document).ready(function(){
                $("img").click(function(){
                    ("h1").text("A Photo of a Breathtaking View");
                });
            });
            ```  

 ### 2. Front-end libraries
   - Boostrap framework
     1. [Starter Template](https://getbootstrap.com/docs/4.0/getting-started/introduction/#starter-template)
     2. [Column Grid Explanation](https://getbootstrap.com/docs/4.0/layout/grid/)
     3. [Containers and Responsive Layout](https://getbootstrap.com/docs/4.0/layout/overview/)
     4. [Images](https://getbootstrap.com/docs/4.0/content/images/)
     5. [Navigation Bars](https://getbootstrap.com/docs/4.0/components/navbar/)
     6. [Font Colors](https://getbootstrap.com/docs/4.0/utilities/colors/)

   - Plotly visualization: Javascript library (Also available in Python)
 
 ### 3. Back-end libraries
   - Flask
     1. Seeing app in the workspace: <code> env | grep WORK </code>
        * `env`: a command which lists all the environmental variables
        
        * `|`: a pipe for sending output from one command to another
        
        * `grep WORK`: command searches text 'WORK'
        
        * Returning result
        ```html
        WORKSPACEDOMAIN= domain value
        WORKSPACEID= id value
        ```
     
     2. Creating new pages (with an example)
        ```html
        <!-- @app.route(): place the web address (path) inside the parentheses -->
        @app.route('/homepage') 
        def render_the_route():
            <!-- index.html file must go in the 'templates' folder -->
            return render_template('index.html')
        ```
        
        * `@`: for decorators, a shorthand way to input a function into another function
        * `@app.route()`: app is a class (or an instance of a class) and route is a method of that class. Hence a function written underneath @app.route() is going to get passed into the route method. The purpose of @app.route() is to make sure the correct web address gets associated with the correct html template.
        * The code ensuring that the web address 'www.some_website.com/homepage` is associated with the index.html template.
     
     3.  Flask with Plotly and Pandas
            ```html
            data = data_wrangling()

            @app.route('/')
            @app.route('/index')
            def index():
                return render_template('index.html', data_set = data)
            ```     
        
        * `data_set = data`: data is sent to the front end via a variable called data_set, which makes the data available to the front_end in the data_set variable. In the `index.html`, the *data_set* variable is accessible using following syntax `{{ data_set }}`. <br>
        Example: 
            ```html
            {% for tuple in data_set %} <!-- One squiggly bracket for control statements -->
                <p>{{tuple}}</p>        <!-- Two squiggly brackets for variables -->
            {% end_for %}
            ```

            ```html
            @app.route('/')
            @app.route('/index')
            def index():
                figures = return_figures()

                # plot ids for the html id tag
                ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

                # Convert the plotly figures to JSON for javascript in html template
                figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

                return render_template('index.html', ids=ids, figuresJSON=figuresJSON)
            ```
        
        * Python is used to set up a Plotly visualization. An id is created associated with each visualization. <br> 
          The Python Plotly code, which is a dictionary of dictionaries, is converted to JSON. The ids and JSON are sent to the front end (index.html). The front end then uses the ids, JSON, and JavaScript Plotly library to render the plots.

   - Deployment
     1. Service providers: [Heroku (or Salesforce.com)](https://www.salesforce.com/), 
                            [Amazon's Lightsail](https://aws.amazon.com/lightsail/), 
                            [Microsoft's Azure](https://azure.microsoft.com/en-us/resources/samples/python-docs-hello-world/), 
                            [Google Cloud](https://cloud.google.com/appengine/docs/standard/python/getting-started/python-standard-env), and 
                            [IBM Cloud (formerly IBM Bluemix)](https://www.ibm.com/blogs/bluemix/2015/03/simple-hello-world-python-app-using-flask/)
     
     2. Code snippet:
        * Step 1. Set an virtual environment and install heroku
            ```html
            <!-- Move all of the web app folders and files into web app folder -->
            mkdir web_app
            mv -t web_app data all_files_and_folders_you_need (eg. worldbankapp wrangling_scripts worldbank.py)

            <!-- Create and activate a virtual environment -->
            conda update python
            python3 -m venv virtual_env_name
            source virtual_env_name/bin/activate

            <!-- Pip install the Python libraries needed for the web app -->
            pip install flask pandas plotly gunicorn

            <!-- Install the heroku command line tools -->
            curl https://cli-assets.heroku.com/install-ubuntu.sh | sh https://devcenter.heroku.com/articles/heroku-cli#standalone-installation
            heroku —-version

            <!-- Log into herokus -->
            heroku login
            ```

        * Step 2. Update worldbank.py or initializing python file
            ```html
            <!-- Before -->
            from worldbankapp import app
            app.run(host='0.0.0.0', port=3001, debug=True)

            <!-- After -->
            from worldbankapp import app
            ```
        
        * Step 3. Create a proc file, which tells Heroku what to do when starting your web app
            ```html
            touch Procfile
            ```
            In the Procfile, type `web gunicorn worldbank(or initializing python file name):app`
        
        * Step 4. Create a requirements file, which lists all of the Python library that your app depends on
            ```html
            pip freeze > requirements.txt
            ```

        * Step 5. Initialize a git repository, configure the email and user name, and make a commit
            ```html
            git init
            git add .
            git config --global user.email email@example.com
            git config --global user.name "my name"
            git commit -m ‘first commit’

            <!-- To check if a remote repository was added to your git repository -->
            git remote -v
            ```

        * Step 6. Create a heroku app and push your git repository to the remote heroku repository
            ```html
            heroku create my-app-name (unique name)
            git push heroku master
            ```

## Part 8-4: Portfolio Exercise - Deploy a Data Dashboard
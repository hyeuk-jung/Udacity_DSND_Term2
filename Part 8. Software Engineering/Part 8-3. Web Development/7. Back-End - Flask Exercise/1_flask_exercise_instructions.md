# Flask Exercise Instructions

You'll find all the files you need for this exercise in the 1_flask_exercise folder. Your task is to add a new route to the routes.py file and a new html file for this route. 

From the 1_flask_exercise/worldbankapp folder, open the routes.py file and the templates folder. Follow the instructions in the TODOS. Once you've changed the files, follow these instructions to see the web app:
1. Open a Terminal
2. Enter the command `env | grep WORK` to find your workspace variables
3. Enter the command `cd 1_flask_exercise` to go into the 1_flask_exercise folder.
4. Enter the command `python worldbank.py`
5. Open a new web browser window and go to the web address:
`http://WORKSPACESPACEID-3001.WORKSPACEDOMAIN` replacing WORKSPACEID and WORKSPACEDOMAIN with your values.

In the workspace, you'll also see a folder called solutions. Inside the solutions folder is another folder called 1_flask_exercise, which contains an example solution.

# Example Result
   * Browser: 
    ```html
        http://viewc7f399f2-3001.udacity-student-workspaces.com
        http://viewc7f399f2-3001.udacity-student-workspaces.com/project_one
        http://viewc7f399f2-3001.udacity-student-workspaces.com/project-one
        http://viewc7f399f2-3001.udacity-student-workspaces.com/project-two
    ```
   * Terminal: 
    ```html
        root@43a96d9d04f7:/home/workspace/1_flask_exercise# python worldbank.py
        * Running on http://0.0.0.0:3001/ (Press CTRL+C to quit)
        * Restarting with stat
        * Debugger is active!
        * Debugger PIN: 291-862-946
        172.18.0.1 - - [25/Jan/2020 06:01:42] "GET / HTTP/1.1" 200 -
        172.18.0.1 - - [25/Jan/2020 06:01:57] "GET /project_one HTTP/1.1" 404 -
        172.18.0.1 - - [25/Jan/2020 06:02:08] "GET /project-one HTTP/1.1" 200 -
        172.18.0.1 - - [25/Jan/2020 06:02:13] "GET /project-two HTTP/1.1" 200 -
```
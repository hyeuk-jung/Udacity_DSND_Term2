# Project: Write a Data Science Blog Post
> Goal: Use the CRISP-DM process to create a blog and Github post <br>


## Table of Contents
  1. [Installation](#installation)  
  2. [Project Motivation](#motivation)  
  3. [File Descriptions](#files)  
  4. [Results](#results)  
  5. [Licensing, Authors, and Acknowledgements](#credits) 


<a id='installation'></a>

## 1. Installation
There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.*.


<a id='motivation'></a>

## 2. Project Motivation
Each year Stack Overflow field a survey covering everything from developers' favorite technologies to their job preference. As a woman and a recent graduate, I was interested in women's current status as a developer and what is the desired features to developers. For this project, I was interested in using Stack Overflow data from 2019 to better understand:

  1. What is the age distribution of a developer for each gender?
     * Variables: `Age` and `Gender`

  2. What is the salary difference between the two genders?
     * Based on the length of professional experience
     * Based on the level of education
     * Variables: `ConvertedComp`, `Gender`, `YearsCodePro`, and `EdLevel` 
  
  3. What are the languages that are expected to be used in 2020?
     * Variable: `LanguageDesireNextYear`
     * The same method can be applied to `DatabaseDesireNextYear`, `PlatformDesireNextYear`, and `WebFrameDesireNextYearplatforms`

For the simplicity, I focused on fully-employed developers' survey results from top 20 countries. 


<a id='files'></a>

## 3. File Descriptions
  ```
  - stack_overflow_survey_2019_analysis.ipynb

  - developer_survey_2019.zip  # Original dataset from Stack Overflow
  
  - developer_survey_2019
  |- README_2019.txt # Description and license information of the dataset
  |- so_survey_2019.pdf # survey form
  |- survey_results_public.csv  # main survey results, one respondent per row and one column per answer
  |- survey_results_schema.csv  # urvey schema, i.e., the questions that correspond to each column name m

  - README.md
  ```


<a id='results'></a>

## 4. Results
The main findings of the code can be found at the post available [here]().

For the summary: 
  1. Male developers outnumber female developers of all ages. So when we compare two genders based on the ratio within each gender, we can observe that the distribution of both sexes is similar in general. Also, about 63% of female developers have age between 20 and 30, and about 54% for male developers, which results in lower average age in female developers.

  2. When comparing the salary of two genders based on experience, we could see the increasing trend for male developers. However, it is hard to observe any pattern for female developers. In terms of education level, both genders show a similar trend in salary though female developers' compensation is lower for all levels of education degree.

  3. In 2020, `C, R, Java, JavaScript, Python, HTML/CSS, SQL, TypeScript, C#, and Bach/Shell/PowerShell` are the top 10 languages desired to be used.


<a id='credits'></a>

## 5. Licensing, Authors, and Acknowledgements
  1. Survey data and license information: [Stack Overflow](https://insights.stackoverflow.com/survey/)


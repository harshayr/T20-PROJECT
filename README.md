<h1>T20 SCORE PREDICTOR</h1>

## ABOUT PROJECT

Introducing t20 matche score predictor

This project is usefull in predicting the score of t20 matches with the help of some parameters such as location, bowling team, batting team, wicketes, run scored in las five overs, current score ,overs done. For predicting the score basic criteria is to atleats 5 overs must have done then you will be able to predict the score 
The most challenging task was the data extraction it took almost 80% of time of project. In this process we extracted data from yaml files and created new features according to our requirment   

Drawbacks:
 * You need atleast last five overs data so atleast five over must have completed
 * You can select only teams , location which are avalaible in dropdown list 
 * It will not able to predict the unusual condictions Ex. 4-5 sixes in a over 2-3 wickets in as over these are abnormal events so model will give wrong prediction in these cases

## How to Install and Run the Project

Step1: First clone the repository using
```sh
git clone https://github.com/harshayr/T20-PROJECT.git
```

Step2: go into current working directory 
```sh
cd T20-PROJECT
```

Step3: Install prerequisites by pasting below command to your terminal
```sh
pip install -r requirment.txt
```

Step4: Run streamlit file using terminal or command promt
```sh
streamlit run main.py
```



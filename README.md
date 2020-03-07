# Disaster Response Pipeline Project
The purpose of the project is to run a flask web app which can classify disaster messages
into certain categories to be able to send them to the correct disaster relief agencies. A multioutput 
randomforrest model was trained on historical messages and is used for prediction.

### Files
* data/disaster_messages.csv- This is the raw message data sent from disaster event
* data/disaster_categories.csv- These are the category types of the disaster messages
* data/process_data.py - This reads data from both the disaster_categories.csv and disaster_messages.csv,
  combines them, cleans them, and writes the data to the DisasterResponse.db
* models/train_classifier.py - This takes data from the DisasterResponse.db, trains a MultioutputClassifier
  random forrest model, and pickles the model to be used later
* model/classifier.pikl - This is the pickled MultioutputClassifier random forrest model
* app/run.py - This starts the flask app which can be opened following the instructions below.

### Required Python Libraries
* sklearn
* numpy
* pandas
* sqlalchemy
* nltk

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

# Disaster Response Pipeline

### Table of Contents

1. [Project overview](#overview)
2. [Requirements](#requirements)
6. [Instructions](#instructions)
5. [Licensing and Acknowledgements](#licensing)

### Project overview:<a name="overview"></a>

- `project-app` contains all python scripts, database and trained classifier,
- `project-etl-ml` contains jupyter notebooks with all steps to process data and train models.

### Requirements:<a name="requirements"></a>

- `pandas`
- `pickle`
- `nltk`
- `sklearn`
- `sqlalchemy`

### Instructions:<a name="instructions"></a>

1. Go to project-app directory
```
cd project-app
```

2. Run the following commands to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://0.0.0.0:3001/

## Licensing and Acknowledgements<a name="licensing"></a>:

Credits must be given to Udacity for the starter codes and FigureEight for provding the data used by this project.


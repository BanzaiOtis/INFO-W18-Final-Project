# INFO-W18-Final-Project
Jan Forslow, Jason Harville, Dan Wald

### Description
This repository contains key files for the final group project in the Spring
2016 semester of INFO W18 for the Berkeley MIDS Program. The merged data sets
on geomagnetic flux in the earth's atmosphere with surface temperature data
and data on electrical production, consumption, and losses in the continental
United States. Details are available in the final report pdf and in its
associated Jupyter Notebook lab book.

### Data
The amount of data necessary to produce the combined data set for this project
could not be stored on GitHub. Instead, this repository contains a single
master csv file of the aggregated data. To access the bulk raw data and view
project drafts, notes, and old code, read-only access is available in this
Google Drive folder:

https://drive.google.com/folderview?id=0BxolO_eh8I-8NTUxUWVTY1Fha0E&usp=sharing

### Code
Due to the need for external data storage, code from this repository will only
work if it is based on the imported master data csv or is related to the
automated downloading of data. To run any code for processing and aggregating
raw data, you must save the source_data directory from the Google Drive link
above and place it in the same directory as the code is being executed from.
It should be clear when functions apply to the raw data.

Additionally, this repository contains a Jupyter Notebook called Analysis
Template, which serves as a blank slate to import the required modules (and
  the associated project_functions.py) and load the master data csv into a
  Pandas dataframe. You must select quick_import=True for this to function if
  you have not downloaded the source_data directory from Google Drive. From this
  point you will be able to explore the data yourself.

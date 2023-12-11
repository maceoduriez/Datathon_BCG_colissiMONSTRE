# Datathon BCG - Equipe ColissiMONSTRE
***
This repo contains everything needed to make predictions on traffic in Paris from 8 to 12 december 2023. Status update : project is still ongoing

## Table of Contents
1. [General Info](#general-info)
2. [Structure](#structure)
3. [Method](#method)
4. [Team composition](#team-composition)

## General Info
***
This project is part of a BCG X datathon that aims to provide a full solution to a delivery company that wants to optimize its delivery process in Paris. In order to do so we first try to predict the traffic to be able to give a data drive solution to the client. The scope of the study is limited to Champs-Elysées avenue, Convention street and Saint-Pères street regarding the short timeline.

## Structure
***
* The [delivery](datathon_bcg/delivery) folder contains the final notebook used to get the results and also the csv file with the results themselves.
* The [module](datathon_bcg/module) folder contains every functions needed to run [the result notebook](datathon_bcg/delivery/final_notebook.ipynb)
* The [data](datathon_bcg/data) folder is old, its first goal was to store the data extracted from [open data website](https://opendata.paris.fr/pages/catalogue/?disjunctive.theme&disjunctive.publisher&sort=modified) but as we wanted to get the most recent data, we wrote function to directly get data from the API

## Method
***
To run our predictions, we use the Meta open-source project Prophet. You can find its documentation [there](https://facebook.github.io/prophet/docs/installation.html). It aims to predict time-series and it is particularly efficient with time series that are driven by strong seasonality effects, both daily, monthly and yearly.


## Team composition
***
Our team is exclusively composed by MEng student from CentraleSupélec. We are all in gap year doing internships in several fields, from finance to consulting or tech. Here is the composition : 
* Macéo Duriez
* Louis Sallé-Tourne
* Lucas Henneçon
* Amine Rakib
* Pierre Pelissier
* Clovis Piedallu


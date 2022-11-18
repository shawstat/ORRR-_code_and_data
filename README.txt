Our code example consists of five parts:

1. The files functions_simu.py and functions_real.py contain all the basic functions we need to use in the simulation and real application, respectively, These functions include:
- SGD_RRR: the realizations of the algorithm ORRR ;
- para_init_: the initialization algorithm ;
- data_gen: the generation of the response and covariate;
- OSMM: the realizations of the algorithm OSAA (proposed in Yang et al.2020);
- err_para: the calculation of the estimation error of parameters in each time point;
- err_pred: the calculation of the prediction error in each time point on the testing set;
- err_regret: the calculation of the mean estimation error at each time point.


2 . The file main_simu.py gives an example showing how to implement the algorithm at different batch sizes m and compare the results. 
If you want to compare other parameters, the program framework is the same. Just fix the other parameters and let the target parameters vary. 
The adjustable parameters are listed in the code including: 
- the batch size (m)
- the dimension of the response (p) and the dimension of the covariate (q)
- the rank of the coefficient matrix (r)
- the noise level (sigma)

3. The file real_app_UKPS.py shows the real application for the UK production-and-sale data.
The original data is publicly available on the OECD website (https://stats.oecd.org/). 
We collected and reorganized the data we were interested in and put it in the file myUKdata.csv. 

4. The file real_app_covid.py shows the real application for the US COVID-19 fatality rate by state.
We used information gathered and made available to the public by the New York Times (https://www.nytimes.com/article/coronavirus-county-data-us.html) on the total number of infections and deaths in each state in the US. 
Obtaining the most recent revision of the current data is simple with the help of the R package covid19nytimes.
We collect the data in the folder coviddata.


5. The file Demo_of_ORRR.ipynb provides a demonstration of our code in the Jupyter notebook.



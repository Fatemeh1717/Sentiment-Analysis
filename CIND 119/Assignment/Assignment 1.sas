/* Assignment 1 */
/* 1 */
/* a */

proc import 
out=breast_cancer
datafile= "V:\breast-cancer.csv"
dbms =csv replace; 
getnames=yes;

proc print data= breast_cancer (obs=10 );

/* b */

ods graphics on;

proc hpsplit data=breast_cancer seed=1 ;
class class age menopause tumor_size inv_nodes node_caps deg_malig breast breast_quad irradiat ;
Model class = age menopause tumor_size inv_nodes node_caps breast breast_quad irradiat deg_malig ;
grow entropy;
prune costcomplexity;
run;

/* 3 */
proc hpsplit data=breast_cancer seed=1 ;
class class age menopause tumor_size inv_nodes node_caps deg_malig breast breast_quad irradiat ;
Model class = age menopause tumor_size inv_nodes node_caps breast breast_quad irradiat deg_malig ;
grow gini;
prune costcomplexity;
run;

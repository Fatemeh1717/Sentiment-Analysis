
/* Lab 1 answers */
/* 1 */
proc import 
out = auto_csv
datafile= "V:\intro_auto.csv"
dbms= csv replace;
getnames=yes;


 proc print data=auto_csv;

 proc contants data=auto_csv;

 run;
/* 2) */

 proc means data=auto_csv;
 run;

 proc freq data=auto_csv;
 tables make foreign repairs;
 run;

 proc univariate data=auto_csv;
 var price mpg;
 run;
 
 /* 3 */
 
 proc boxplot data=auto_csv;
 plot price*make;
 run;

/* 4 */

 proc import
 out=auto_noisy_csv
 datafile="V:\intro_auto_noisy.csv"
 dbms=csv replace;
 getnames=yes;


 proc print data=auto_noisy_csv;
 data auto_clean;
 set auto_noisy_csv;

 
proc sort data=auto_clean out=auto_clean;
by price mpg;
run;

proc print data=auto_clean;
run;

proc sort data=auto_clean out=auto_clean;
by mpg price;
run;
data auto_clean;
set auto_clean;
by price mpg;
if first.price;
if first.mpg;
run;
proc print data= auto_clean;
run;


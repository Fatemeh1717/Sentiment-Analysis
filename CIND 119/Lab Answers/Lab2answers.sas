/* Lab 2*/
/* 1*/

proc import 
out=mtcars
datafile="V:\mtcars.csv"
dbms=csv replace;
getnames=yes;

proc print data=mtcars;

proc ttest data=mtcars sides=2 alpha=0.05;
title "t-test:Miles Per Gallon(mpg)";
class am;
var mpg;
run;

/* 2*/
proc import 
out= sleep
datafile= "V:\sleep.csv"
dbms=csv replace;
getnames=yes;
 proc print data=sleep;

 proc ttest data=sleep sides=2 alpha=0.1 h0=7;
 title "one sample t-tes example";
 var sleep;
 run;


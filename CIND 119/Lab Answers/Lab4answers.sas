/* Lab 4 */
/* 1 */
/* a */
proc import 
out=iris
datafile="V:\Lab 4 iris.csv"
dbms=csv replace;
getnames=yes;

proc print data= iris(obs=10);
title 'IRIS DATABASE'
run;

/* b */

proc sgplot;
scatter x=sepal_length y=petal_length/ group=variety;
title"IRIS DATABASE";
run;

/* 2 */
/* a */
proc stdize method= std out=iris_stdize;
var sepal_length sepal_width petal_length petal_width;
run;

proc print data= iris_stdize(obs=10);

/* b */
%macro doFASTCLUS;
%do k=2 %to 5;
title 'CLUSTER in IRIS with '&k'-Means';


proc fastclus
data=iris_stdize out=iris_cluster
maxiter=100 maxclusters=&k
summary;

proc sgplot;
scatter x=sepal_length y=petal_length/ datalabel=cluster  group=variety;
run;
%end;
%mend;
%doFASTCLUS


PROC print data=iris_cluster;
run;

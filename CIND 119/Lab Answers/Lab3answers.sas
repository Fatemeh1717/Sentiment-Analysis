/* Lab 3 */
/* 1 */
/* a */

proc import
out= wine
datafile="V:\wine.csv"
dbms=csv replace;
getnames=yes;


proc print data=wine(obs=10);
title"Wine Database";
run;

/* b, c */

ods graphics on;

proc hpsplit data=wine;
class WineType;
model WineType= Alcohol Malic_acid Ash Alcalinity_of_ash Magnesium Total_phenols
Flavanoids Nonflavanoid_phenols Proanthocyanins Color_intensity Hue OD280_OD315 Proline;
grow entropy;
prune costcomplexity;
run;

/* 2 */
/* a */
 proc import 
 out=mtcars
 datafile="V:\mtcars.csv"
 dbms=csv replace;
 getnames=yes;

 proc print data=mtcars;
 title "mpg regression tree";
 run;

 ods graphics on;

 proc hpsplit data=mtcars maxdepth=10 seed=1;
 class cyl am gear vs carb;
 model mpg= cyl am gear vs carb hp disp drat qsec wt;
 grow VARIANCE;
 prune none;
 runn;







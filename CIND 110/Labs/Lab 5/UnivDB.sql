create database UnivDb;
use UnivDb;

Create table Student ( Id int primary key,
						Name varchar(40),
                        Major varchar(40),
                        Dept varchar(40)
                        
);

insert into Student values ( 16 , 'Jack' , 'Bioengineering' , 'CS'),
							(17 , 'Ryan' , 'Mechatronics' , 'MIE'),
							(18 , 'Sally' , 'Data Science' , 'CS'),
							(19 , 'Jane' , 'Software Engineering' , 'MIE')

;

SET FOREIGN_KEY_CHECKS=0; 

Create table Course (
						Name char(20),
                        Code char(20) primary key,
                        Credit int,
                        Dept varchar(40)
                        
);

insert into Course values  ('Data Structures' , 'CCPS305' , 3 , 'CS'),
						('Data Organization' , 'CIND110' , 4 , 'MIE'),
						('Data Analytics' , 'CIND123' , 2 , 'MIE'),
						('Python Programming' , 'CIND830' , 3 , 'MIE')


;

Create table Section ( Id char(20)primary key,
						C_Code char(20),
                        Term varchar(40),
                        Yera int,
                        Instructor varchar(40),
                        foreign key (C_Code) references Course(Code)
);

insert into Section values ( 'YJ5' , 'CIND123' , 'Spring' , 2020 , 'Sally'),
							('KJ2' , 'CCPS305' , 'Fall' , 2021 , 'King'),
							('YJ2' , 'CIND110' , 'Winter' , 2019 , 'Larry'),
							('YJ3' , 'CIND110' , 'Fall' , 2020 , 'Sandy'),
							('KJ3' , 'CIND110' , 'Winter' , 2019 , 'King')


;


Create table Grade ( Std_Id int ,
						Sec_Id char(20),
                        Pct_Grade varchar(20),
                        Ltr_Grade varchar(20),
                        primary key( Std_Id, Sec_Id),
                        foreign key( Std_Id)references Student(Id),
                        foreign key( Sec_Id)references Section(Id)
						
                        
);

insert into Grade values (16 , 'YJ2' , 74 , 'B'),
							(17 , 'YJ2' , 75 , 'B'),
							(18 , 'YJ3' , 65 , 'C'),
							(17 , 'KJ3' , 88 , 'A'),
							(19 , 'YJ3' , 74 , 'B')
;

alter table section
rename column Yera to Year;

# The following example retrieves all the information of the sections conducted after the year of 2019

select distinct * 
from section
where Year>2019;

# The following example retrieves all the information of the sections conducted after 2019 during the Fall term.

select distinct * 
from section
where Year>2019 and Term='Fall';

select distinct * 
from section
where  Term='Fall' and Year>2019;

#The following example retrieves all the information of computer science or industrial engineering 
#sections conducted during the Fall term

SELECT distinct * 
from section
where C_Code like 'CIND110%' and Term = 'Fall' or C_Code like 'CCPS%' and Term = 'Fall';


SELECT distinct * 
from section
where C_Code like 'CIND110%' OR  C_Code like 'CCPS%' 
				AND Term = 'Fall' ;



# The following example retrieves all the names of the instructors in the SECTION table

select distinct Instructor
from section;

# The following example renames the COURSE relation before filtering the rows and then choosing a subset of attributes.

select C.Code, C.Name , C.Dept
from course AS C
where credit <>3;

# The following example lists all the records in both of the Name attribute from the
# STUDENT relation and the Instructor attribute from the SECTION relation.alter

select distinct instructor from section
union
select distinct Name from student;

# The following example lists the intersection between the records in the Name attribute from the
# STUDENT relation and the Instructor attribute from the SECTION relation, displaying only
# those who are both students and instructors.

select distinct instructor from section
where	 Instructor in(
select distinct Name from student);

# The following example lists the difference between the records in the Name attribute from the STUDENT relation and the
# Instructor attribute from the SECTION relation, displaying only those who are instructors but not students.

select distinct instructor from section
where	 Instructor not in(
select distinct Name from student);

# The following example shows the result of applying the CROSS PRODUCT
# operation on two relations that are not union-compatible.
select  distinct instructor , Dept
from course cross join section;

# The following example shows the number of the resulting rows after cross joining all the
 # four relations in our dataset.
 
 select  distinct *  
from student cross join section cross join course cross join grade ;
 
 # Assume we want to retrieve the name of the student of each listed grade
 select distinct * 
 from student as S inner join grade as G 
 ON s.Id = G.Std_Id;
 
 # The following example shows the difference between applying the LEFT OUTER JOIN and the Right OUTER JOIN (./ ) 
# operation on the STUDENT and SECTION relations.

select distinct S.Name , K.instructor
from student as S LEFT OUTER JOIN section AS K
ON S.Name = K.Instructor;


select distinct S.Name , K.instructor
from student as S right OUTER JOIN section AS K
ON S.Name = K.Instructor;

# The following example shows how to retrieve the number of courses in each department using the COUNT() aggregate function, which is used to count records or values
select dept, count(code) as Count_Course
from course
group by dept;

# The following example extends the previous one by adding a conditional expression to filter
 #the groups using the HAVING clause before displaying the resulting attributes.
 select dept, count(code) as Count_Course
from course
group by dept
having Count_Course >= 3;
 


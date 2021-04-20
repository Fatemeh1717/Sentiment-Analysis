
-- 1)

create table test_data(
ID integer Not null IDENTITY(1,1),
class varchar(50) Not Null,
age integer,
menopause varchar(200),
tumor_size integer,
node_caps varchar(50) ,
deg_malig integer,
brest varchar(100),
breast_quad varchar(200),
irradiat varchar(50)
primary key (ID));

-- 2)

insert into test_data(class,age,menopause,tumor_size,node_caps,deg_malig,brest,breast_quad,irradiat) Values ('NO',35,'premeno',31,'no',3,'left','left_low','no')
insert into test_data(class,age,menopause,tumor_size,node_caps,deg_malig,brest,breast_quad,irradiat) Values ('NO',42,'premeno',22,'no',2,'right','right_up','no')
insert into test_data(class,age,menopause,tumor_size,node_caps,deg_malig,brest,breast_quad,irradiat) Values ('NO',30,'premeno',23,'no',2,'left','left_low','no')
insert into test_data(class,age,menopause,tumor_size,node_caps,deg_malig,brest,breast_quad,irradiat) Values ('NO',61,'ge40',16,'no',2,'right','left_up','no')
insert into test_data(class,age,menopause,tumor_size,node_caps,deg_malig,brest,breast_quad,irradiat) Values ('NO',45,'premeno',2,'no',2,'right','right_low','no')
insert into test_data(class,age,menopause,tumor_size,node_caps,deg_malig,brest,breast_quad,irradiat) Values ('NO',64,'ge40',17,'no',2,'left','left_low','no')
insert into test_data(class,age,menopause,tumor_size,node_caps,deg_malig,brest,breast_quad,irradiat) Values ('NO',52,'premeno',27,'no',2,'left','left_low','no')
insert into test_data(class,age,menopause,tumor_size,node_caps,deg_malig,brest,breast_quad,irradiat) Values ('NO',67,'ge40',21,'no',1,'left','left_low','no')
insert into test_data(class,age,menopause,tumor_size,node_caps,deg_malig,brest,breast_quad,irradiat) Values ('YES',41,'premeno',52,'no',2,'left','left_low','no')
insert into test_data(class,age,menopause,tumor_size,node_caps,deg_malig,brest,breast_quad,irradiat) Values ('YES',43,'premeno',22,'no',2,'right','left_up','no')
insert into test_data(class,age,menopause,tumor_size,node_caps,deg_malig,brest,breast_quad,irradiat) Values ('YES',41,'premeno',1,'no',3,'left','central','no')
insert into test_data(class,age,menopause,tumor_size,node_caps,deg_malig,brest,breast_quad,irradiat) Values ('YES',44,'ge40',27,'no',2,'left','left_low','no')
insert into test_data(class,age,menopause,tumor_size,node_caps,deg_malig,brest,breast_quad,irradiat) Values ('YES',61,'It40',14,'no',1,'left','right-up','no')
insert into test_data(class,age,menopause,tumor_size,node_caps,deg_malig,brest,breast_quad,irradiat) Values ('YES',55,'ge40',26,'no',3,'left','right-up','no')
insert into test_data(class,age,menopause,tumor_size,node_caps,deg_malig,brest,breast_quad,irradiat) Values ('YES',44,'premeno',32,'no',3,'left','left_up','no')



-- 3-a)


select * from test_data
where menopause='ge40';

-- 3-b)

select * from test_data
where age <41;

-- 3-c) There is not any record for this Query

select * from test_data
where age <41 and menopause='ge40';


-- 3-d) 

select AVG(age) from test_data;

-- 3-e)

select AVG(age) from test_data
where deg_malig=3;










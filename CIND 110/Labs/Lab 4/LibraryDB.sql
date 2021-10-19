drop database librarydb;
CREATE database LibraryDB;



create table Publisher( Name char(60) primary key,
						Address varchar(60),
                        Phone varchar(60));
                        
                        
INSERT INTO Publisher VALUES ('AEI Press',NULL,NULL),
							('Coach House Books','Toronto, Canada','111-111-2345'),
							('Currency Press','Redfern, New South Wales, Australia',NULL),
                            ('HarperCollins','New York, New York, United States','222-222-2222'),
                            ('John Wiley & Sons','Hoboken, New Jersey, United States',NULL),
                            ('Jonathan Cape','London, United Kingdom','555-555-5555'),
                            ('Random House','Random House Tower, New York City, United States',NULL),
                            ('Scribner','New York, United States','333-333-3333'),
                            ('Turnstone Press','Canada',NULL),
                            ('UIT',NULL,NULL),
                            ('Vintage Books','New York, New York, United States','666-666-6666');
					
create table Book ( Book_id int primary key,
					Title varchar(255),
                    Publisher_Name char(60),
                   constraint Fk_publisher_name foreign key (Publisher_Name)
                    references Publisher(Name)


);                        
SET foreign_key_checks = 0;
INSERT INTO BOOK VALUES (1,'Life 3.0: Being Human in the Age of Artificial Intelligence','Vintage Books'),
(2,'Should We Eat Meat?: Evolution and Consequences of Modern Carnivory','John Wiley & Sons'),
(3,'I Contain Multitudes: The Microbes Within Us and a Grander View of Life','HarperCollins'),
(4,'Behind the Beautiful Forevers: Life, Death, and Hope in a Mumbai Undercity','Random House'),
(5,'The Most Powerful Idea in the World: A Story of Steam, Industry, and Invention','Jonathan Cape'),
(6,'Sustainable Energy - Without the Hot Air','UIT'),
(7,'The Emperor of All Maladies: A Biography of Cancer','Scribner'),
(8,'The Gene: An Intimate History','Scribner'),
(9,'Energy Myths and Realities: Bringing Science to the Energy Policy Debate','AEI Press'),
(10,'Prepared: What Kids Need for a Fulfilled Life','Currency Press');

DROP TABLE Book_Authors;
create table Book_Authors ( Book_id int,
					
                    Author_Name varchar(60) ,
                    primary key(Book_id,Author_Name),
					constraint Fk_book_book_id foreign key (Book_id)
                    references Book(Book_id)

);  
INSERT INTO Book_Authors VALUES (1,'Max Tegmark '),
(2,'Vaclav Smil'),(3,'Ed Yong'),(4,'Katherine Boo'),(5,'William Rosen'),
(6,'David MacKay'),
(7,'Siddhartha Mukherjee'),
(8,'Siddhartha Mukherjee'),
(9,'Vaclav Smil'),
(10,'Diane Tavenner');

create table Borrower ( Card_no int primary key,
					Name varchar(60),
                    Address varchar(60),
                    Phone varchar(60));

INSERT INTO Borrower VALUES (166,'Carmen J Clyburn','2604 Victoria Park Ave, ON',NULL),
							(432,'Sythoun Sun','3094 Brew Creek Road, Whistler, BC','543-234-8765'),
                            (500,'Ali Ahmadi','3853 Broadmoor Blvd, Mississauga, ON',NULL),
                            (543,'Diane Tavenner','3094 Brew Creek Road, Whistler, BC','345-542-7777'),
                            (703,'Sythoun Sun','3783 Front Street, Toronto, Ontario','321-654-1987'),
                            (777,'Stainbrook Green','1065 Glover Road, Hannon, ON',NULL),
                            (816,'Jess Huynh','3094 Brew Creek Road, Whistler, BC','123-456-7891'),
                            (976,'Harry V. Stainbrook','1065 Glover Road, Hannon, ON','321-456-7891'),
                            (978,'Ethel V. Stainbrook','1074 Glover Road, Hannon, ON','654-321-1987');
                            
create table Book_Loans ( Book_id int ,
						Due_date Date,
                        Due_out date,
                        Branch_id int,
                        Card_no int,
                        primary key(Book_id,Branch_id,Card_no),
                        constraint Fk_borrower_card_no foreign key (Card_no)
						references Borrower(Card_no),
                        constraint Fk_book_bookid foreign key (Book_id)
						references Book(Book_id),
						constraint Fk_LibraryBranch_Branch_id foreign key (Branch_id)
						references Library_Branch(Branch_id)
                        
                        
);  

INSERT INTO Book_Loans VALUES (1,'2020-02-10','2020-02-01',3025,166),
							(3,'2020-04-03','2020-02-01',3025,166),
                            (4,'2020-01-01','2020-02-05',3568,976),
                            (4,'2020-01-29','2020-02-01',4156,703),
                            (5,'2020-01-14','2020-01-03',5489,816),
                            (6,'2020-03-01','2020-02-01',3025,703),
                            (6,'2020-01-15','2019-12-26',3025,976),
                            (6,'2020-03-01','2020-02-06',3568,976);
                            
create table Book_Copies( Book_id int,
                        Branch_id int,
                        No_of_copies int,
                        primary key (Book_id,Branch_id),
                        constraint Fk_LibraryBranch_Branchid foreign key (Branch_id)
						references Library_Branch(Branch_id)
                        
);  

INSERT INTO Book_Copies VALUES (1,3025,10),
								(2,3025,30),
								(2,5489,7),
                                (3,3025,70),
                                (3,4156,11),
                                (4,4156,15),
                                (5,3568,80),
                                (6,5489,60),
                                (7,3025,NULL),
                                (7,5489,10),
                                (8,3211,30),
                                (8,3568,40),
                                (9,3025,44),
                                (9,3568,0),
                                (10,3568,50),
                                (10,4156,60);
create table Library_Branch ( 
                        Branch_id int primary key,
                        Branch_Name varchar(60),
                        Address varchar(60)
); 

insert INTO Library_Branch VALUES (3025,'Albert Campbell','496 Birchmount Road Toronto, ON  M1K 1N8'),
(3211,'Barbara Frum','20 Covington Rd, North York, ON M6A 3C1'),
(3444,'Champlain Heights','7110 Kerr St, Vancouver, BC V5S 4W2'),
(3568,'Bayview','2901 Bayview Ave, North York, ON M2K 1E6'),
(4156,'Northern District','40 Orchard View Blvd, Toronto, ON M4R 1B9'),
(4567,'Central Saanich','1209 Clarke Rd, Brentwood Bay, BC V8M 1P8'),
(5489,'Albion','1515 Albion Road Toronto, ON  M9V 1B2 '); 


#1) The following query retrieves all borrowers who do not have a phone number.

select * 
from borrower
where phone is null;

#2)- The following example retrieves the title of all books with a publisher whose name starts with
# the letter ’J’ or their phone number does not begin with ’333’.

select Title, publisher_name
from Book as B 
where  B.publisher_name IN ( select P.name 
							from Publisher as P
							where P.name like 'J% ' OR
                            P.phone NOT like '333%');
# Second form
select Title, publisher_name
from Book as B
join Publisher as P 
on B.publisher_name = P.Name
where P.name like 'J% ' OR P.phone NOT like '333%';

# if we try to retrieve each author’s name similar to the borrower’s name, we can use the following correlated queries
 select Author_name 
 from Book_Authors As BA
 where BA.Author_name in( select BR.Name
							from Borrower as BR
                            Where BA.Author_name = BR.name);
 
 #  The following example retrieves the names of the publishers who have no books stored in our database.

select P.name 
from publisher AS P
where  not exists (select *
						from Book as B
                        where P.name = B.publisher_name);
 # we would like to retrieve the library branch id and the borrower card number of every book with more than ten copies
 Select  DISTINCT BC.Branch_id  , BL.Card_no 
 from book_loans as BL 
 jOIN book_copies AS BC
 ON BL.Branch_id = BC. Branch_id
 WHERE BC.No_of_copies >10;

Select  DISTINCT BC.Branch_id  , BL.Card_no 
 from book_loans as BL 
 nATURAL JOIN  book_copies AS BC
 WHERE BC.No_of_copies >10;
 
 
 Select  DISTINCT BC.Branch_id  , BL.Card_no 
 from book_loans as BL 
JOIN  book_copies AS BC
USING (Branch_id, Book_id)
 WHERE BC.No_of_copies >10;
 
 # retrive  only authors who are borrowers are included in the results .
  select BA.Author_Name , BR.name
  from book_authors AS BA 
  JOIN borrower AS BR
  ON 
  BA.Author_Name = BR.Name;
  
 # If we need to include all the authors in the results, a different type of join, called OUTER JOIN should be considered.
 
 select BA.Author_Name , BR.name
  from book_authors AS BA 
  LEFT outer join borrower AS BR
  ON 
  BA.Author_Name = BR.Name;
  
  
  # Similarly, if we are interested in displaying the list of all the borrowers in the BORROWER table, we can use the RIGHT OUTER JOIN
 
 select BA.Author_Name , BR.name
  from book_authors AS BA 
  right outer join borrower AS BR
  ON 
  BA.Author_Name = BR.Name;
 
 #  The CROSS JOIN operation is used to specify the Cartesian Product operation resulting in all possible record combinations.

select BA.Author_Name , BR.name
  from book_authors AS BA 
  cross join borrower AS BR;
  
  # The following example computes the sum of the copies of all books in each library branch
  select Branch_id, sum(No_of_copies) as 'Total Number of copy'
  from Book_Copies
  Group by Branch_id;
  
  # The following example displays the average number of books per branch that is below 30.
  
  select Branch_id, avg(No_of_copies) as 'Average Number of copy'
  from Book_Copies
  Group by Branch_id
  having 'Average Number of copy'<30;
  
 # The following example retrieves the branch id and the total number of copies more than 20 for each branch located in Ontario. 
 
  select Branch_id, sum(No_of_copies) as 'Total Number of copy'
  from Book_Copies
  where Branch_id in
					(select Branch_id
                    from library_branch
                    where Address like '%ON%')
group by Branch_id;
 
# to add a constraint to the LibraryDB database stating that the number of copies of any book must not be less than 5, we can design the following trigger

CREATE trigger Copies_Violation 
			before insert on book_copies
            for each row
            set new.No_of_copies = if (new.No_of_copies <5 ,
											Null,
                                            new.No_of_copies);
                                            
 select * from book_copies; 
 show triggers;
 
INSERT INTO book_copies VALUES(7, 3028, 4); 
INSERT INTO BOOK_COPIES VALUES(9, 3029, 44);
 
 #The following example shows how to create a virtual table, following the LibraryDB database schema.
 create view Available_copies as
				select * 
                from book_copies
                where No_of_copies between 20 and 30
                order by No_of_copies desc;
                
show create view Available_copies;
 
-- MySQL Workbench Synchronization
-- Generated: 2021-09-21 17:08
-- Model: New Model
-- Version: 1.0
-- Project: Name of the project
-- Author: Asma

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

ALTER TABLE `rookerydb`.`bird` 
DROP FOREIGN KEY `FK_Family_Sci_Name`;

ALTER TABLE `rookerydb`.`bird_family` 
DROP FOREIGN KEY `FK_Order_Sci_Name`;

ALTER TABLE `rookerydb`.`bird` 
DROP COLUMN `Bird_Status`,
DROP COLUMN `Nail_Beak`,
DROP COLUMN `Body_Id`,
DROP COLUMN `Last_Seen_dt`,
CHANGE COLUMN `Common_Name` `Common_Name` VARCHAR(50) NULL DEFAULT NULL ,
DROP INDEX `FK_Family_Sci_Name` ;
;

ALTER TABLE `rookerydb`.`bird_family` 
DROP INDEX `FK_Order_Sci_Name` ;
;

DROP TABLE IF EXISTS `rookerydb`.`bird_copy` ;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;

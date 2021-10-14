-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema mydb
-- -----------------------------------------------------
-- -----------------------------------------------------
-- Schema insurance
-- -----------------------------------------------------
-- -----------------------------------------------------
-- Schema rookerydb
-- -----------------------------------------------------

-- -----------------------------------------------------
-- Schema rookerydb
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `rookerydb` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci ;
USE `rookerydb` ;

-- -----------------------------------------------------
-- Table `rookerydb`.`bird_order`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `rookerydb`.`bird_order` (
  `order_id` INT NOT NULL AUTO_INCREMENT,
  `scientific_name` VARCHAR(255) NULL DEFAULT NULL,
  `brief_description` TEXT NULL DEFAULT NULL,
  PRIMARY KEY (`order_id`),
  UNIQUE INDEX `scientific_name` (`scientific_name` ASC) VISIBLE)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `rookerydb`.`bird_family`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `rookerydb`.`bird_family` (
  `Family_id` INT NOT NULL AUTO_INCREMENT,
  `Scientific_name` VARCHAR(255) NULL DEFAULT NULL,
  `brief_description` TEXT NULL DEFAULT NULL,
  `order_sci_name` VARCHAR(255) NULL DEFAULT NULL,
  PRIMARY KEY (`Family_id`),
  UNIQUE INDEX `Scientific_name` (`Scientific_name` ASC) VISIBLE,
  CONSTRAINT `fk_order_sci_name`
    FOREIGN KEY (`order_sci_name`)
    REFERENCES `rookerydb`.`bird_order` (`scientific_name`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `rookerydb`.`bird`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `rookerydb`.`bird` (
  `Bird_id` INT NOT NULL AUTO_INCREMENT,
  `Scientific_name` VARCHAR(255) NULL DEFAULT NULL,
  `Common_name` VARCHAR(255) NULL DEFAULT NULL,
  `Family_sci_name` VARCHAR(255) NULL DEFAULT NULL,
  `Brief_description` TEXT NULL DEFAULT NULL,
  `Last_seen_dt` DATETIME NULL DEFAULT NULL,
  `Body_id` CHAR(2) NULL DEFAULT NULL,
  `Nail_beak` TINYINT(1) NULL DEFAULT NULL,
  `Bird_image` BLOB NULL DEFAULT NULL,
  `Bird_status` ENUM('Accidental', 'Breeding', 'Extinct', 'Extirpated', 'Introduced') NULL DEFAULT NULL,
  PRIMARY KEY (`Bird_id`),
  UNIQUE INDEX `Scientific_name` (`Scientific_name` ASC) VISIBLE,
  CONSTRAINT `fk_family_sci_name`
    FOREIGN KEY (`Family_sci_name`)
    REFERENCES `rookerydb`.`bird_family` (`Scientific_name`))
ENGINE = InnoDB
AUTO_INCREMENT = 8
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `rookerydb`.`bird_copy`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `rookerydb`.`bird_copy` (
  `Bird_id` INT NOT NULL AUTO_INCREMENT,
  `Scientific_name` VARCHAR(255) NULL DEFAULT NULL,
  `Common_name` VARCHAR(255) NULL DEFAULT NULL,
  `Family_sci_name` VARCHAR(255) NULL DEFAULT NULL,
  `Brief_description` TEXT NULL DEFAULT NULL,
  `Last_seen_dt` DATETIME NULL DEFAULT NULL,
  `Body_id` CHAR(2) NULL DEFAULT NULL,
  `Nail_beak` TINYINT(1) NULL DEFAULT NULL,
  `Bird_image` BLOB NULL DEFAULT NULL,
  `Bird_status` ENUM('Accidental', 'Breeding', 'Extinct', 'Extirpated', 'Introduced') NULL DEFAULT NULL,
  PRIMARY KEY (`Bird_id`),
  UNIQUE INDEX `Scientific_name` (`Scientific_name` ASC) VISIBLE)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `rookerydb`.`bird_deep_copy`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `rookerydb`.`bird_deep_copy` (
  `Bird_id` INT NOT NULL DEFAULT '0',
  `Scientific_name` VARCHAR(255) NULL DEFAULT NULL,
  `Common_name` VARCHAR(255) NULL DEFAULT NULL,
  `Family_sci_name` VARCHAR(255) NULL DEFAULT NULL,
  `Brief_description` TEXT NULL DEFAULT NULL,
  `Last_seen_dt` DATETIME NULL DEFAULT NULL,
  `Body_id` CHAR(2) NULL DEFAULT NULL,
  `Nail_beak` TINYINT(1) NULL DEFAULT NULL,
  `Bird_image` BLOB NULL DEFAULT NULL,
  `Bird_status` ENUM('Accidental', 'Breeding', 'Extinct', 'Extirpated', 'Introduced') NULL DEFAULT NULL)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;

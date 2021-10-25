-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema mydb
-- -----------------------------------------------------
-- -----------------------------------------------------
-- Schema healthinsurancedb
-- -----------------------------------------------------

-- -----------------------------------------------------
-- Schema healthinsurancedb
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `healthinsurancedb` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci ;
USE `healthinsurancedb` ;

-- -----------------------------------------------------
-- Table `healthinsurancedb`.`agent`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `healthinsurancedb`.`agent` (
  `SIN` CHAR(9) NOT NULL,
  `First_Name` VARCHAR(40) NULL DEFAULT NULL,
  `Last_Name` VARCHAR(40) NULL DEFAULT NULL,
  `Speciality` VARCHAR(50) NULL DEFAULT NULL,
  `Experience` INT NULL DEFAULT NULL,
  PRIMARY KEY (`SIN`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `healthinsurancedb`.`client`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `healthinsurancedb`.`client` (
  `HealthCardNo` CHAR(12) NOT NULL,
  `First_Name` VARCHAR(40) NULL DEFAULT NULL,
  `Last_Name` VARCHAR(40) NULL DEFAULT NULL,
  `Address` VARCHAR(100) NULL DEFAULT NULL,
  `Age` INT NULL DEFAULT NULL,
  PRIMARY KEY (`HealthCardNo`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `healthinsurancedb`.`clientagent`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `healthinsurancedb`.`clientagent` (
  `Client_HealthCardNo` CHAR(12) NOT NULL,
  `Agent_SIN` CHAR(9) NOT NULL,
  PRIMARY KEY (`Client_HealthCardNo`, `Agent_SIN`),
  INDEX `Fk_Agent_SIN` (`Agent_SIN` ASC) VISIBLE,
  INDEX `Fk_Client_HealthCardNo` (`Client_HealthCardNo` ASC) VISIBLE,
  CONSTRAINT `Fk_Agent_SIN`
    FOREIGN KEY (`Agent_SIN`)
    REFERENCES `healthinsurancedb`.`agent` (`SIN`),
  CONSTRAINT `Fk_Client_HealthCardNo`
    FOREIGN KEY (`Client_HealthCardNo`)
    REFERENCES `healthinsurancedb`.`client` (`HealthCardNo`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `healthinsurancedb`.`contracts`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `healthinsurancedb`.`contracts` (
  `StartDate` DATE NOT NULL,
  `EndDate` DATE NOT NULL,
  `Text` VARCHAR(512) NOT NULL,
  PRIMARY KEY (`StartDate`, `EndDate`, `Text`),
  INDEX `EndDate` (`EndDate` ASC) VISIBLE,
  INDEX `Text` (`Text` ASC) INVISIBLE)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `healthinsurancedb`.`healthpolicyissuers`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `healthinsurancedb`.`healthpolicyissuers` (
  `Name` CHAR(100) NOT NULL,
  `PhoneNo` VARCHAR(15) NOT NULL,
  PRIMARY KEY (`Name`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `healthinsurancedb`.`provincialhealthdep`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `healthinsurancedb`.`provincialhealthdep` (
  `Name` CHAR(100) NOT NULL,
  `PhoneNo` VARCHAR(15) NOT NULL,
  `Address` VARCHAR(100) NOT NULL,
  `Price` INT NULL DEFAULT NULL,
  `Curency` INT NULL DEFAULT NULL,
  PRIMARY KEY (`Name`, `PhoneNo`, `Address`),
  INDEX `PhoneNo` (`PhoneNo` ASC) INVISIBLE,
  INDEX `Address` (`Address` ASC) VISIBLE)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `healthinsurancedb`.`phdcontract`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `healthinsurancedb`.`phdcontract` (
  `PHDName` CHAR(100) NOT NULL,
  `PHDPhoneNo` VARCHAR(15) NOT NULL,
  `PHDAddress` VARCHAR(100) NOT NULL,
  `ContractStartDate` DATE NOT NULL,
  `ContractEndDate` DATE NOT NULL,
  `ContractText` VARCHAR(512) NOT NULL,
  PRIMARY KEY (`PHDName`, `PHDPhoneNo`, `PHDAddress`, `ContractStartDate`, `ContractEndDate`, `ContractText`),
  INDEX `Fk_PHDContractPhoneNo` (`PHDPhoneNo` ASC) VISIBLE,
  INDEX `Fk_PHDContractAddress` (`PHDAddress` ASC) VISIBLE,
  INDEX `Fk_Contracts_StartDate` (`ContractStartDate` ASC) VISIBLE,
  INDEX `Fk_Contracts_EndDate` (`ContractEndDate` ASC) VISIBLE,
  INDEX `Fk_Contracts_Text` (`ContractText` ASC) VISIBLE,
  INDEX `Fk_PHDContractName` (`PHDName` ASC) VISIBLE,
  CONSTRAINT `Fk_Contracts_EndDate`
    FOREIGN KEY (`ContractEndDate`)
    REFERENCES `healthinsurancedb`.`contracts` (`EndDate`),
  CONSTRAINT `Fk_Contracts_StartDate`
    FOREIGN KEY (`ContractStartDate`)
    REFERENCES `healthinsurancedb`.`contracts` (`StartDate`),
  CONSTRAINT `Fk_Contracts_Text`
    FOREIGN KEY (`ContractText`)
    REFERENCES `healthinsurancedb`.`contracts` (`Text`),
  CONSTRAINT `Fk_PHDContractAddress`
    FOREIGN KEY (`PHDAddress`)
    REFERENCES `healthinsurancedb`.`provincialhealthdep` (`Address`),
  CONSTRAINT `Fk_PHDContractName`
    FOREIGN KEY (`PHDName`)
    REFERENCES `healthinsurancedb`.`provincialhealthdep` (`Name`),
  CONSTRAINT `Fk_PHDContractPhoneNo`
    FOREIGN KEY (`PHDPhoneNo`)
    REFERENCES `healthinsurancedb`.`provincialhealthdep` (`PhoneNo`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `healthinsurancedb`.`policy`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `healthinsurancedb`.`policy` (
  `Trade_Name` CHAR(100) NOT NULL,
  `Symbol` VARCHAR(50) NULL DEFAULT NULL,
  PRIMARY KEY (`Trade_Name`),
  UNIQUE INDEX `Trade_Name` (`Trade_Name` ASC) VISIBLE,
  INDEX `Fk_HPI_Name` (`Trade_Name` ASC) VISIBLE,
  CONSTRAINT `Fk_HPI_Name`
    FOREIGN KEY (`Trade_Name`)
    REFERENCES `healthinsurancedb`.`healthpolicyissuers` (`Name`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `healthinsurancedb`.`phdpolicy`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `healthinsurancedb`.`phdpolicy` (
  `PHDName` CHAR(100) NOT NULL,
  `PHDPhoneNo` VARCHAR(15) NOT NULL,
  `PHDAddress` VARCHAR(100) NOT NULL,
  `PolicyTrade_Name` CHAR(100) NOT NULL,
  PRIMARY KEY (`PHDName`, `PHDPhoneNo`, `PHDAddress`, `PolicyTrade_Name`),
  UNIQUE INDEX `PolicyTrade_Name` (`PolicyTrade_Name` ASC) VISIBLE,
  INDEX `Fk_PHDPhoneNo` (`PHDPhoneNo` ASC) VISIBLE,
  INDEX `Fk_PHDAddress` (`PHDAddress` ASC) VISIBLE,
  INDEX `Fk_PHDName` (`PHDName` ASC) VISIBLE,
  INDEX `Fk_PolicyTrade_Name` (`PolicyTrade_Name` ASC) VISIBLE,
  CONSTRAINT `Fk_PHDAddress`
    FOREIGN KEY (`PHDAddress`)
    REFERENCES `healthinsurancedb`.`provincialhealthdep` (`Address`),
  CONSTRAINT `Fk_PHDName`
    FOREIGN KEY (`PHDName`)
    REFERENCES `healthinsurancedb`.`provincialhealthdep` (`Name`),
  CONSTRAINT `Fk_PHDPhoneNo`
    FOREIGN KEY (`PHDPhoneNo`)
    REFERENCES `healthinsurancedb`.`provincialhealthdep` (`PhoneNo`),
  CONSTRAINT `Fk_PolicyTrade_Name`
    FOREIGN KEY (`PolicyTrade_Name`)
    REFERENCES `healthinsurancedb`.`policy` (`Trade_Name`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `healthinsurancedb`.`recommendations`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `healthinsurancedb`.`recommendations` (
  `Date` DATE NOT NULL,
  `Quantity` CHAR(12) NOT NULL,
  PRIMARY KEY (`Date`, `Quantity`),
  INDEX `Fk_Policy_Trade_Name` (`Quantity` ASC) VISIBLE,
  INDEX `Fk_ReClient_HealthCardNo` (`Quantity` ASC) VISIBLE,
  CONSTRAINT `Fk_Policy_Trade_Name`
    FOREIGN KEY (`Quantity`)
    REFERENCES `healthinsurancedb`.`policy` (`Trade_Name`),
  CONSTRAINT `Fk_ReClient_HealthCardNo`
    FOREIGN KEY (`Quantity`)
    REFERENCES `healthinsurancedb`.`client` (`HealthCardNo`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;

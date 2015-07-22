#!/bin/bash

mkdir -p historical-data/output
cd historical-data
wget https://resources.lendingclub.com/LoanStats3a.csv.zip
wget https://resources.lendingclub.com/LoanStats3b.csv.zip
wget https://resources.lendingclub.com/LoanStats3c.csv.zip
wget https://resources.lendingclub.com/LoanStats3d.csv.zip
unzip LoanStats3a.csv.zip
unzip LoanStats3b.csv.zip
unzip LoanStats3c.csv.zip
unzip LoanStats3d.csv.zip
rm -Rf *.zip


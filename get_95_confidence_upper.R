#!/usr/bin/Rscript

args <- commandArgs(TRUE)
df <- read.csv(args[1], header=FALSE)
names(df) <-c("energy")
mean(df$energy)+(1.96 * sd(df$energy))

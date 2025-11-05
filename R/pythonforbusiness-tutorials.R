#
# Autor : Bernd Heesen
# Skript: Tutorials zu Paket "pythonforbusiness"
#

##---Pakete laden--------------------------------------------------------------
{
  packages <- c("devtools","roxygen2","tidyverse","learnr","reticulate")
  lapply(packages,library, character.only = TRUE, warn.conflicts = FALSE)
  rm(packages)
} 

##---Paket pythonforbusiness laden---------------------------------------------
{
# !!Hinweis: Die "#" in Zeile 16+17 entfernen, um das Paket zu laden  
# devtools::install_github("bheesen/pythonforbusiness", force = TRUE)
# library(pythonforbusiness)
}

##---Paket testen--------------------------------------------------------------
  ml.tutorial(name = "py.syntax")
  ml.tutorial(name = "py.datenstrukturen")
  ml.tutorial(name = "py.operationen")
  ml.tutorial(name = "py.kontrollstrukturen")
  ml.tutorial(name = "py.funktionen")
  ml.tutorial(name = "py.standardfunktionen")
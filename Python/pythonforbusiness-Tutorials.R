#
# Autor : Bernd Heesen
# Skript: Tutorials zu Paket "pythonforbusiness"
#

##---Paket machinelearning laden-----------------------------------------------
{
# Paket machinelearning laden
# devtools::install_github("bheesen/pythonforbusiness", force = TRUE)
  library(pythonforbusiness)
# help(package = "pythonforbusiness")
}

##---Paket testen--------------------------------------------------------------
  ml.tutorial(name = "py.syntax")
  ml.tutorial(name = "py.datenstrukturen")
  ml.tutorial(name = "py.operationen")
  ml.tutorial(name = "py.kontrollstrukturen")
  ml.tutorial(name = "py.funktionen")
  ml.tutorial(name = "py.standardfunktionen")
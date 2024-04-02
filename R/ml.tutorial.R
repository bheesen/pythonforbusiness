#' Function to start a tutorial accompanying the package "pythonforbusiness" and the book "Künstliche Intelligenz for Business mit Python".
#'
#' @param name is the name of the tutorial
#' @keywords tutorial
#' @example  ms.tutorial(name = "ml.syntax")
#' @export

ml.tutorial <-  function(name){
  learnr::run_tutorial(name, package = "pythonforbusiness")
}  

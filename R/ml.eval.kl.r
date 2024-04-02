#' Function to evaluate the quality of a prediction for categorical variables (classification)
#'
#' @param df is a dataframe with two variables: real values, estimated values
#' @param titel is the name of the variable to be used as subtitle on diagrams
#' @param xvar name of the column of the true variable in df
#' @param yvar name of the column of the estimated variable df
#' @param scatter if TRUE a scatter-chart is created
#' @keywords machine learning, modeling, evaluation, classification
#' @return auc
#' @example ml.eval.lr(autos.eval[,c("Preis","Vorhersage")],"Gebrauchtwagen","realer Preis","vorhergesagter Preis",scatter=T)
#' @export
ml.eval.kl<-function (df,titel="",xvar="",yvar="",scatter=F) 
  # df:             Dataframe mit 2 Variablen, 1.Wahrheit, 2.Schätzwerte
  # titel:          Untertitel 
  # xvar,yvar:      Variablennamen/Spaltennamen in df
  # scatter:        TRUE=Scatter-Chart visualisieren        
{
  var.auc<-NA
  df<-df[,c(xvar,yvar)]
  colnames(df)<-c("x","y")
  df$y<-as.numeric(df$y)
  kz.kl<-metric_set(accuracy,mcc,f_meas)          # Kennzahlen für Klassifikation
  kz.kl(two_class_example,truth=truth,estimate=predicted)
  var.auc<-roc_auc(df,x,y)$.estimate
  if (scatter==TRUE){
    roc.werte <- roc_curve(df,x,y)                # ROC-Werte für y
    p.scatter<-ggplot(roc.werte[,c("specificity","sensitivity")])+                          
      aes(x=1-specificity,y=sensitivity)+
      geom_point(alpha=0.1)+
      geom_abline(lty=1)+
      labs(title="Evaluation des Klassifkations-Modells",
           subtitle=paste0(titel," (AUC= ",round(var.auc,2),")"),
           x="Spezifität",y="Sensitivität")
    grid.arrange(p.scatter,nrow=1,ncol=1)
    var.ev<-paste("Modellevaluations-Plot fehlerfrei")
  }
  return(var.auc)
}

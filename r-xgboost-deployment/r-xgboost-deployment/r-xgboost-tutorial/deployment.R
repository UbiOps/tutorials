library(caret)
library(xgboost)

#' @title Init
#' @description Initialisation method for the deployment.
#' It can for example be used for loading modules that have to be kept in memory or setting up connections.
init <- function(base_directory, context) {
print("Initialising My Deployment")
model <- file.path(base_directory, "xgb_fit.RDS")
print("Loading model")
xgb_fit <<- readRDS(model)
print("Model succesfully loaded")

}

#' @title Request
#' @description Method for deployment requests, called separately for each individual request.
#' @return output data (str or named list) request result
#'     - In case of structured output: a named list, with as keys the output fields as defined upon deployment creation
#'     - In case of plain output: a string
request <- function(input_data, base_directory, context){
  print("Reading CSV")
  data <- read.csv(input_data[["input_data"]], stringsAsFactors = FALSE)

  print("Making predictions")

  predictions <- predict(xgb_fit, data, "raw")

  output_path <- file.path(base_directory, "predictions.csv")
  write.csv(predictions, output_path, row.names = FALSE)
  
return(list("prediction" = output_path))
  }

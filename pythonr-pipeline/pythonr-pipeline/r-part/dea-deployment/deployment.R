#' @title Init
#' @description Initialisation method for the deployment.
#'     It can for example be used for loading modules that have to be kept in memory or setting up connections.

init <- function(base_directory, context) {
  print("Initialising My Deployment")
}

#' @title Request
#' @description Method for deployment requests, called separately for each individual request.
#' @return output data (str or named list) request result
#'     - In case of structured output: a named list, with as keys the output fields as defined upon deployment creation
#'     - In case of plain output: a string
request <- function(input_data, base_directory, context) {
 print("Processing request for My Deployment")

 # Load in the required libraries
 library(ggplot2)
 library(GGally)
 library(gridExtra)
 library(scales)
  
 df <- read.csv(input_data[["raw_data"]], stringsAsFactors = FALSE)
 # Remove the date column
 df <- subset(df, select = -c(date))


 condition <- df$condition
 grade <- df$grade
 price <- df$price
  
 print(paste("rows:", nrow(df), "cols:", ncol(df)))

 # Lets look at the distribution of house condtion, grade and price:
 p1 <- qplot(condition, data = df, geom = "bar",
              main = "Number of houses by condition")

 p2  <- qplot(grade, data = df, geom = "bar",
              main = "Number of houses by grade")

 p3 <- ggplot(df, aes(x = price)) + geom_density(stat = "density") +
    xlab("price")  + ggtitle("Price distribution") 


  
 # Look at price (log10) vs other features:
 ggplot(df, aes(x = log10(price), y = sqft_living)) +
  geom_smooth() +
  scale_y_continuous(labels = comma) +
  scale_x_continuous(labels = comma) +
  ylab("sqft of living area") +
  geom_point(shape = 1, alpha = 1/10) +
  ggtitle("Price (log10) vs sqft of living area")

 ggplot(df, aes(x = grade, y = log10(price))) +
  geom_boxplot() +
  scale_y_continuous(labels = comma) +
  coord_flip() +
  geom_point(shape = 1, alpha = 1/10) +
  ggtitle("Price (log10) vs grade")
  

 ggplot(df, aes(x = condition, y = log10(price))) +
  geom_boxplot() +
  scale_y_continuous(labels = comma) +
  coord_flip() +
  geom_point(shape = 1, alpha = 1/10) +
  ggtitle("Price (log10) vs condition")
  

 ggplot(df, aes(x = as.factor(floors), y = log10(price))) +
  geom_boxplot() +
  scale_y_continuous(labels = comma) +
  xlab("floors") +
  coord_flip() +
  geom_point(shape = 1, alpha = 1/10) +
  ggtitle("Price (log10) vs number of floors")
 
 
 # Look at how the different features correlate
  ggcorr(df, hjust = 0.8, layout.exp = 1) +
  ggtitle("Correlation between house features")
    

 # Create a DataFrame that only contains the columns which we will use to predict the house prices
 data <- df[, c('sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms', 'view', 'sqft_basement', 'lat', 'waterfront', 'yr_built', 'bedrooms')]

 output_path <- file.path(getwd(), "data.csv")
 write.csv(data, output_path, row.names = FALSE)
  
 return(list("clean_data" = output_path))
  
}
head(data)


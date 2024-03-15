# Install and load in the required packages
library(caret)
library(xgboost)

# Set the working directory to the file containing the r-xgboost-deployment
setwd("<INSERT_YOUR_PATH_HERE>")

# Read the data into a dataframe and prepare the data for the prediction
data <- read.csv("kc_house_data.csv", header = TRUE)

# Remove the id and date column
data <- data[-c(1, 2)]

# Split the data
train_idx <- createDataPartition(data$price, p = .85, list = FALSE)
train <- data[train_idx, ]
test <- data[-train_idx, ]

# Extract the true values from the test set
test_labels <- test[, 1]

# Set up the resampling method
ctrl <- trainControl(method = "cv", number = 10, allowParallel = TRUE)

# Set the parameters
param_grid <- expand.grid(eta = c(0.3),
                          max_depth = c(6),
                          gamma = c(0),
                          colsample_bytree = c(0.6),
                          nrounds = c(120),
                          min_child_weight = c(1),
                          subsample = 1)

# Train the model
xgb_fit <- train(price ~ .,
            data = data, method = "xgbTree", metric = "RMSE",
            trControl = ctrl, subset = train_idx, tuneGrid = param_grid)

# Make the predicitons
predictions <- predict(xgb_fit, test, "raw")

# Lets see how the prediction model scored
sum <- summary(predictions)
cor <- (cor(predictions, test_labels))
rmse <- RMSE(predictions, test_labels)

print(paste(sum))
print(paste("The accuracy is: ", cor))
print(paste("The RMSE is: ", rmse))

# Save the model
saveRDS(xgb_fit, "r-xgboost/xgb_fit.RDS")

# Download the UbiOps client library if necessary
install.packages("devtools")
library(devtools)
install_github("UbiOps/client-library-r")
library(ubiops)
library(utils)

# Zip the file
zip(zipfile = "r-xgboost", files = "r-xgboost") # Note: this function does not work with Windows 10, please zip the file manually if you are working with that OS.
r_xgboost_zip <- file.path(getwd(), "r-xgboost.zip")  #Make sure the working directory is set to the directory which contains the deployment_package.zip



# Connect to UbiOps environment
Sys.setenv("UBIOPS_PROJECT" = "INSERT_YOUR_PROJECT_NAME_HERE")
Sys.setenv("UBIOPS_API_TOKEN" = "INSERT_YOUR_TOKEN_HERE")


Sys.setenv(UBIOPS_API_URL = "https://api.ubiops.com/v2.1")

DEPLOYMENT_NAME <- "r-xgboost-deployment"
DEPLOYMENT_VERSION <- "v1"

result <- service_status()
result

# Create the R deployment
deployment <- list(
  name = DEPLOYMENT_NAME,
  description = "r-xgboost-deployment.",
  input_type = "structured",
  output_type = "structured",
  input_fields = list(
    list(name = "input_data", data_type = "file")
  ),
  output_fields = list(
    list(name = "prediction", data_type = "file")
  ),
  labels = list(demo = "r-xgboost-deployment")
)
result <- deployments_create(data = deployment)
result

# Create a deployment version
version <- list(
  version = DEPLOYMENT_VERSION,
  environment = "r4-0",
  instance_type = "2048mb",
  maximum_instances = 1,
  minimum_instances = 0,
  maximum_idle_time = 1800, # = 30 minutes
  request_retention_mode = "none"  # We don't need to store the requests in this demo
)

result <- deployment_versions_create(
  deployment.name = DEPLOYMENT_NAME,
  data = version
)
result

# Upload the zipped deployment package
result <- revisions_file_upload(
  deployment.name = DEPLOYMENT_NAME,
  version = DEPLOYMENT_VERSION,
  file = r_xgboost_zip
)
build.id <- result$build
result

status <- "queued"
while(status != "success" && status != "failed") {
  result <- builds_get(
    deployment.name = DEPLOYMENT_NAME,
    version = DEPLOYMENT_VERSION,
    build.id = build.id
  )
  status <- result$status
  Sys.sleep(2)
}
print(status)


result <- deployment_versions_get(
  deployment.name = DEPLOYMENT_NAME,
  version = DEPLOYMENT_VERSION
)
result$status
# Install and load in the required packages
library(caret)
library(xgboost)

# Set the working directory to the file containing the r-xgboost-pipeline
#setwd("<INSERT_YOUR_PATH_HERE>")
setwd("<INSERT_YOUR_PATH_HERE>")

# Read the data into a dataframe and prepare the data for the prediction
raw_data <- read.csv("kc_house_data.csv", header = TRUE)

# Remove the id and date column
raw_data <- raw_data[-c(1, 2)]

# Subset the columns that shall be used for the prediction
data <- raw_data[, c("price", "sqft_living", "grade", "sqft_above", "sqft_living15", "bathrooms", "view", "sqft_basement", "lat", "waterfront", "yr_built", "bedrooms")]

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
saveRDS(xgb_fit, "prediction-deployment/xgb_fit.RDS")

# Download the UbiOps client library if necessary
install.packages("devtools")
library(devtools)
install_github("UbiOps/client-library-r")
library(ubiops)
library(utils)

# Zip the eda-deployment folder
zip(zipfile = "eda-prep-deployment", files = "eda-prep-deployment") # Note: this function does not work with Windows 10, please zip the file manually if you are working with that OS.
eda_zip <- file.path(getwd(), "eda-prep-deployment.zip")  #Make sure the working directory is set to the directory which contains the deployment_package.zip


# Zip the pred-deployment files
zip(zipfile = "prediction-deployment", files = "prediction-deployment") # Note: this function does not work with Windows 10, please zip the file manually if you are working with that OS.
pred_zip <- file.path(getwd(), "prediction-deployment.zip")

# Connect to UbiOps environment
Sys.setenv("UBIOPS_PROJECT" = "INSERT_YOUR_PROJECT_NAME_HERE")
Sys.setenv("UBIOPS_API_TOKEN" = "<INSERT_YOUR_TOKEN_HERE>")


Sys.setenv(UBIOPS_API_URL = "https://api.ubiops.com/v2.1")

DEPLOYMENT_NAME1 <- "eda-prep-deployment"
DEPLOYMENT_NAME2 <- "prediction-deployment"
DEPLOYMENT_VERSION <- "v1"

PIPELINE_NAME = "r-pred-pipeline"
PIPELINE_VERSION = "v1"

result <- service_status()
result

# Create the eda-prep-deployment
deployment <- list(
  name = DEPLOYMENT_NAME1,
  description = "eda-prep-deployment",
  input_type = "structured",
  output_type = "structured",
  input_fields = list(
    list(name = "raw_data", data_type = "file")
  ),
  output_fields = list(
    list(name = "clean_data", data_type = "file")
  ),
  labels = list(demo = "eda-prep-deployment")
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
  request_retention_mode = "none"  # We don"t need to store the requests in this demo
)

result <- deployment_versions_create(
  deployment.name = DEPLOYMENT_NAME1,
  data = version
)
result

# Upload the zipped deployment package
result <- revisions_file_upload(
  deployment.name = DEPLOYMENT_NAME1,
  version = DEPLOYMENT_VERSION,
  file = eda_zip
)
build.id <- result$build
result

# Create the prediction-deployment
deployment <- list(
  name = DEPLOYMENT_NAME2,
  description = "prediction-deployment",
  input_type = "structured",
  output_type = "structured",
  input_fields = list(
    list(name = "clean_data", data_type = "file")
  ),
  output_fields = list(
    list(name = "prediction", data_type = "file")
  ),
  labels = list(demo = "prediction-deployment")
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
  request_retention_mode = "none"  # We don"t need to store the requests in this demo
)

result <- deployment_versions_create(
  deployment.name = DEPLOYMENT_NAME2,
  data = version
)
result

# Upload the zipped deployment package
result <- revisions_file_upload(
  deployment.name = DEPLOYMENT_NAME2,
  version = DEPLOYMENT_VERSION,
  file = pred_zip
)
build.id <- result$build
result
status <- "queued"
while(status != "success" && status != "failed") {
  result <- builds_get(
    deployment.name = DEPLOYMENT_NAME2,
    version = DEPLOYMENT_VERSION,
    build.id = build.id
  )
  status <- result$status
  Sys.sleep(2)
}
print(status)


result <- deployment_versions_get(
  deployment.name = DEPLOYMENT_NAME2,
  version = DEPLOYMENT_VERSION
)
result$status

# Now that the deployments have been build, it is time to create the pipeline 
pipeline <- list(
    name = PIPELINE_NAME,
    description = "Pipeline that predics house prices",
    input_type = "structured",
    input_fields = list(
        list(name = "raw_data", data_type = "file")
    ),
    output_type = "structured",
    output_fields = list(
        list(name = "prediction", data_type = "file")
    ),
    labels = list(demo = "r-pipeline")
)
result_pipe <- pipelines_create(data = pipeline)
result_pipe


pipeline_version <- list(
    version = PIPELINE_VERSION,
    request_retention_mode="none"  # We don"t need to store the requests in this de
)

result <- pipeline_versions_create(
    pipeline.name = PIPELINE_NAME,
    data = pipeline_version
)
result

# Lets add the deployments as objects to the pipeline
object <- list(
    name = DEPLOYMENT_NAME1,
    reference_name = DEPLOYMENT_NAME1,
    version = DEPLOYMENT_VERSION
)
result <- pipeline_version_objects_create(
    pipeline.name = PIPELINE_NAME,
    version = PIPELINE_VERSION,
    data = object
)
result


object <- list(
    name = DEPLOYMENT_NAME2,
    reference_name = DEPLOYMENT_NAME2,
    version = DEPLOYMENT_VERSION
)
result <- pipeline_version_objects_create(
    pipeline.name = PIPELINE_NAME,
    version = PIPELINE_VERSION,
    data = object
)
result

# After both objects have been added to the pipeline, the final step is to connect the components
# pipeline_start -> eda-prep-deployment
attachment <- list(
    destination_name = DEPLOYMENT_NAME1,
    sources = list(
        list(
            source_name = "pipeline_start",
            mapping = list(
                list(source_field_name = "raw_data", destination_field_name = "raw_data")
            )
        )
    )
)
result <- pipeline_version_object_attachments_create(
    pipeline.name = PIPELINE_NAME, 
    version = PIPELINE_VERSION,
    data = attachment
)
result

# eda-prep-deployment -> prediction-deployment
attachment2 <- list(
    destination_name = DEPLOYMENT_NAME2,
    sources = list(
        list(
            source_name = DEPLOYMENT_NAME1,
            mapping = list(
                list(source_field_name = "clean_data", destination_field_name = "clean_data")
            )
        )
    )
)
result <- pipeline_version_object_attachments_create(
    pipeline.name = PIPELINE_NAME, 
    version = PIPELINE_VERSION,
    data = attachment2
)
result

# prediction-deployment -> pipeline_end
attachment <- list(
    destination_name = "pipeline_end",
    sources = list(
        list(
            source_name = DEPLOYMENT_NAME2,
            mapping = list(
                list(source_field_name = "prediction", destination_field_name = "prediction")
            )
        )
    )
)

# Create the version
result <- pipeline_version_object_attachments_create(
    pipeline.name = PIPELINE_NAME, 
    version = PIPELINE_VERSION,
    data = attachment
)
result



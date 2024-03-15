# Download the UbiOps client library if necessary
install.packages("devtools")
library(devtools)
install_github("UbiOps/client-library-r")
library(ubiops)
library(utils)



# Set the working directory to the directory to the r-part folder, within the pythonr-pipeline folder
setwd("<INSERT_PATH_TO_DIRECTORY_HERE>") 

zip(zipfile = "dea-deployment", files = "dea-deployment") # This function doesn't work on Windows 10 machines,
                                                          # please zip the files manually if you are running this script from a Windows 10 machine 



dea_deployment_zip <- file.path(getwd(), "dea-deployment.zip")  #Make sure the working directory is set to the directory which contains the deployment_package.zip


# Connect to UbiOps environment
Sys.setenv("UBIOPS_PROJECT" = "<INSERT_YOUR_PROJECT_HERE>")
Sys.setenv("UBIOPS_API_TOKEN" = "<INSERT_API_TOKEN_WITH_PROJECT_EDITOR_RIGHTS>")


Sys.setenv(UBIOPS_API_URL = "https://api.ubiops.com/v2.1")

DEPLOYMENT_NAME <- "r-eda-deployment"
DEPLOYMENT_VERSION <- "v1"


PIPELINE_NAME <- "pythonr-pipeline"
PIPELINE_VERSION <- "v1"



result <- service_status()
result

# Create the R deployment
deployment <- list(
  name = DEPLOYMENT_NAME,
  description = "r-eda-deployment.",
  input_type = "structured",
  output_type = "structured",
  input_fields = list(
    list(name = "raw_data", data_type = "file")
  ),
  output_fields = list(
    list(name = "clean_data", data_type = "file")
  ),
  labels = list(demo = "r-eda-deployment")
)
result <- deployments_create(data = deployment)
result

# Create a deployment version
version <- list(
  version = DEPLOYMENT_VERSION,
  environment = "r4-0",
  instance_type = "256mb",
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
  file = dea_deployment_zip
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

# After the deployment has been created, return to the Jupyter notebook and run step 5
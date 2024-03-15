
# Enter your project name below
PROJECT_NAME = '<YOUR_PROJECT_NAME>'
DEPLOYMENT_NAME = 'tensorflow-deployment'
DEPLOYMENT_VERSION = 'v1'

import os
import ubiops
from pathlib import Path
import multiprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
import hydra
from hydra.utils import to_absolute_path
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from src.factory import get_model, get_optimizer, get_scheduler
from src.generator import ImageSequence
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf


# Import all necessary libraries
import shutil
import os
import ubiops



#@hydra.main(config_path="age_gender_estimation/src/config.yaml")
#we cannot use this in jupyter notebooks
def main(cfg):
    if cfg.wandb.project:
        import wandb
        from wandb.keras import WandbCallback
        wandb.init(project=cfg.wandb.project)
        callbacks = [WandbCallback()]
    else:
        callbacks = []
        
    data_path = Path("/pfs/faces/data/imdb_crop")
    #data_path = Path("/home/raoulfasel/Documents/pachyderm/age_gender_estimation/data/imdb_crop")
    
    csv_path = Path(to_absolute_path("./")).joinpath("meta", f"{cfg.data.db}.csv")
    #csv_path = Path(to_absolute_path("/pfs/faces")).joinpath("meta", f"{cfg.data.db}.csv")
    print(csv_path)
    df = pd.read_csv(str(csv_path))
    train, val = train_test_split(df, random_state=42, test_size=0.1)
    train_gen = ImageSequence(cfg, train, "train", data_path)
    val_gen = ImageSequence(cfg, val, "val", data_path)

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = get_model(cfg)
        opt = get_optimizer(cfg)
        scheduler = get_scheduler(cfg)
        model.compile(optimizer=opt,
                      loss=["sparse_categorical_crossentropy", "sparse_categorical_crossentropy"],
                      metrics=['accuracy'])

    #checkpoint_dir = Path(to_absolute_path("age_gender_estimation")).joinpath("checkpoint")
    checkpoint_dir = Path(to_absolute_path("/pfs/build")).joinpath("checkpoint")

    print(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    filename = "_".join([cfg.model.model_name,
                         str(cfg.model.img_size),
                         "weights.{epoch:02d}-{val_loss:.2f}.hdf5"])
    callbacks.extend([
        LearningRateScheduler(schedule=scheduler),
        ModelCheckpoint(str(checkpoint_dir) + "/" + filename,
                        monitor="val_loss",
                        verbose=1,
                        save_best_only=True,
                        mode="auto")
    ])

    model.fit(train_gen, epochs=cfg.train.epochs, callbacks=callbacks, validation_data=val_gen,
              workers=multiprocessing.cpu_count())
    
    model.save("tensorflow_deployment_package/tensorflow_model.h5")

    with open('/opt/ubiops/token', 'r') as reader:
        API_TOKEN = reader.read()
    client = ubiops.ApiClient(ubiops.Configuration(api_key={'Authorization': API_TOKEN}, 
                                               host='https://api.ubiops.com/v2.1'))
    api = ubiops.CoreApi(client)
    
    # Create the deployment
    deployment_template = ubiops.DeploymentCreate(
        name=DEPLOYMENT_NAME,
        description='Tensorflow deployment',
        input_type='structured',
        output_type='structured',
        input_fields=[
            ubiops.DeploymentInputFieldCreate(
                name='input_image',
                data_type='file',
            ),
        ],
        output_fields=[
            ubiops.DeploymentOutputFieldCreate(
                name='output_image',
                data_type='file'
            ),
        ],
        labels={"demo": "tensorflow"}
    )

    api.deployments_create(
        project_name=PROJECT_NAME,
        data=deployment_template
    )

    # Create the version
    version_template = ubiops.DeploymentVersionCreate(
        version=DEPLOYMENT_VERSION,
        environment='python3-8',
        instance_type="2048mb",
        minimum_instances=0,
        maximum_instances=1,
        maximum_idle_time=1800 # = 30 minutes
    )

    api.deployment_versions_create(
        project_name=PROJECT_NAME,
        deployment_name=DEPLOYMENT_NAME,
        data=version_template
    )

    # Zip the deployment package
    shutil.make_archive('tensorflow_deployment_package', 'zip', '.', 'tensorflow_deployment_package')

    # Upload the zipped deployment package
    file_upload_result =api.revisions_file_upload(
        project_name=PROJECT_NAME,
        deployment_name=DEPLOYMENT_NAME,
        version=DEPLOYMENT_VERSION,
        file='tensorflow_deployment_package.zip'
    )

if __name__ == '__main__':
    #with initialize(config_path="age_gender_estimation/src/"):
    with initialize(config_path="src/"):
        cfg = compose(config_name="config")
        print(OmegaConf.to_yaml(cfg))
    main(cfg)

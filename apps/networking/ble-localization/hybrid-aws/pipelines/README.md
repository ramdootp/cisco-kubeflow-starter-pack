# BLERSSI Hybrid Pipeline 

# Cisco UCS 🤝 SageMaker

## Pre-requisites

- [ ] UCS machine with Kubeflow installed
- [ ] AWS account with appropriate permissions

## AWS S3 Create Bucket

Ensure you have the AWS CLI installed. 
Otherwise, you can use the docker image with the alias set.

    alias aws='docker run --rm -it -v ~/.aws:/root/.aws -v $(pwd):/aws amazon/aws-cli'
    aws s3 mb s3://mxnet-model-store --region us-west-2

## SageMaker permissions

In order to run this pipeline, we need to prepare an IAM Role to run Sagemaker jobs. You need this `role_arn` to run a pipeline. Check [here](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) for details.

This pipeline also use aws-secret to get access to Sagemaker services, please also make sure you have a `aws-secret` in the kubeflow namespace.

    echo -n $AWS_ACCESS_KEY_ID | base64
    echo -n $AWS_SECRET_ACCESS_KEY | base64

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: aws-secret
  namespace: kubeflow
type: Opaque
data:
  AWS_ACCESS_KEY_ID: YOUR_BASE64_ACCESS_KEY
  AWS_SECRET_ACCESS_KEY: YOUR_BASE64_SECRET_ACCESS
```

## Run notebook to create pipeline

Ensure you have jupyter lab installed on your local machine. And Kubeflow installed 

    git clone https://github.com/CiscoAI/cisco-kubeflow-starter-pack cksp
    cd cksp
    cd apps/networking/ble-localization/hybrid-aws/pipelines/
    
    
### Upload Notebook file

Upload [blerssi-aws.ipynb](blerssi-aws.ipynb) file in jupyter notebook
    
### Run Pipeline

Open the [blerssi-aws.ipynb](blerssi-aws.ipynb) file and run pipeline

Set the input parameters for the pipeline in the first cell of the notebook.

![BLERSSI Pipeline](./pictures/notebook-sabe-1.PNG)

Import libraries and set model and deploy component yaml paths

![BLERSSI Pipeline](./pictures/notebook-sabe-2.PNG)

Define BLERSSI mxnet pipeline function

![BLERSSI Pipeline](./pictures/notebook-sabe-3.PNG)

Create kubeflow experiment with name "BLERSSI-Sagemaker"

![BLERSSI Pipeline](./pictures/notebook-sabe-4.PNG)

### Note :

  Build your inference docker image and push to your ECR account. Please follow the below steps
    
  1. Run the build and push script from [here](./components/v1/mxnet-byom-inference/container/build_and_push.sh) by using your account credentials.

Set AWS region, and inference image to built ECR image

![BLERSSI Pipeline](./pictures/notebook-sabe-5.PNG)

Create BLERSSI run and open run link

![BLERSSI Pipeline](./pictures/notebook-sabe-6.PNG)


The BLERSSI Sagemaker pipeline starts execting. 
Once all the components executed successfully, check the logs of sagemaker-deploy component to verify endpoint is created.

![BLERSSI Pipeline](./pictures/notebook-sabe-7.PNG)

To verify endpoint in AWS, open AWS sagemaker and check endpoints created successfully as snapshot given below

![BLERSSI Pipeline](./pictures/aws-sagemaker-endpoint.PNG)

### Run Prediction API

To predict the output go back to jupyter notebook and start executing other cells

Provide AWS credentials

![BLERSSI Pipeline](./pictures/notebook-sabe-8.PNG)

Predicted result will be displayed

![BLERSSI Pipeline](./pictures/notebook-sabe-9.PNG)

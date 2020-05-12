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

Open the `blerssi-aws.ipynb` notebook.
Set the input parameters for the pipeline in the first cell of the notebook.
Here are the parameters and samples for each of them.

> Note! Set the bucket name from the first step

```
Parameters:
[Required]
bucket_name (string): S3 bucket to be used by the pipeline
    Sample: "mxnet-model-store"

[Required]
role_arn (string): SageMaker Role ARN for execution of pipeline components
    Sample: 'arn:aws:iam::${account_id}:role/service-role/AmazonSageMaker-ExecutionRole-${timestemp}'

execution_mode (string): where the notebook is being run
    Sample: 'in-cluster' (default), 'local' 

aws_secret_name (string): AWS secret where IAM creds are stored
    Sample: 'aws-secret' (default)

[Optional]
host (string): KF Pipelines service endpoint
    Sample:  "http://10.10.10.10:31380/pipeline"
```

#!/usr/bin/env python
# coding: utf-8

# # Tutorial : Building an AWS SageMaker End-to-end Workflow with BentoML
# 
# # Introduction
# 
# ### Description: 
# 
# This tutorial provides an end-to-end guide using BentoML with AWS SageMaker -- a machine learning model training platform. It demonstrates the workflow of integrating BentoML with SageMaker, including: setting up a SageMaker notebook instance, model training, creating an S3 bucket, uploading the BentoService bundle into S3, and deploying the BentoML packaged model to SageMaker as an API endpoint using the BentoML CLI tool. 
# 
# For demonstration, this tutorial uses the IMDB movie review sentiment dataset with BERT and Tensorflow 2.0.(please note: the following model is a modification of the original version: https://github.com/kpe/bert-for-tf2/blob/master/examples/gpu_movie_reviews.ipynb)
# 
# ### Objectives: 
# 
# - demonstrates an end-to-end workflow of using BentoML with AWS SageMaker
# - deploys and tests API endpoints to the cloud 
# - provides two ways of API server local testing using BentoML CLI tool
# 

# ## 1. Create a SageMaker notebook instance 
# 
# For model training in SageMaker, we will start by creating a **notebook instance**. After logging into the AWS management console -- type SageMaker to launch the service. From the SageMaker dashboard, select Notebook instances. Go ahead enter a notebook name and select the instance type 
# 
# ![Screen%20Shot%202020-06-26%20at%203.42.37%20PM.png](attachment:Screen%20Shot%202020-06-26%20at%203.42.37%20PM.png)
# 
# Next,under **Permissions and encryption**, select **Create a new role** or **choosing an existing role**. This allows both the notebook instance and user to access and upload data to Amazon S3. Then, select Any S3 bucket, which allows your SageMaker to access all S3 buckets in your account.
# 
# ![Screen%20Shot%202020-06-24%20at%2012.13.02%20PM.png](attachment:Screen%20Shot%202020-06-24%20at%2012.13.02%20PM.png)
# 
# After the notebook instance is created, the status will change from pending to **InService**. Select Open Jupyter under Actions, and choose **Conda_python 3** under New tab to launch the Jupyter notebook within SageMaker. 
# 
# Note: SageMaker also provides a local model through pip install SageMaker.
# 
# ![Screen%20Shot%202020-06-24%20at%2012.47.34%20PM.png](attachment:Screen%20Shot%202020-06-24%20at%2012.47.34%20PM.png)
# 
# 

# Finally to prepare for the model training, let's import some libraries -- Boto3 and SageMaker and set up the IAM role. Boto3 is the AWS SDK for Python, which makes it easier to integrate our model with AWS services such as Amazon S3

# In[ ]:


import boto3, sagemaker
from sagemaker import get_execution_role

# Define IAM role
role = get_execution_role()
prefix = 'sagemaker/bert-moviereview-bento'
my_region = boto3.session.Session().region_name # set the region of the instance


# 

# In this step, we will create an S3 bucket named movie-review-dataset to store the dataset. Users could click on the bucket name and upload the dataset directly into S3. Alternatively, for cost-efficiency, users could train the model locally using the SageMaker local mode

# In[7]:


bucket_name = 'movie-review-dataset'
s3 = boto3.resource('s3')
s3.create_bucket(Bucket=bucket_name)


# ![Screen%20Shot%202020-06-24%20at%204.26.33%20PM.png](attachment:Screen%20Shot%202020-06-24%20at%204.26.33%20PM.png)

# ## 2. Model Training -- Movie review sentiment with BERT and TensorFlow2
# 
# The second step of this tutorial is model training. We will be using the IMDB movie review dataset to create a sentiment analysis model which contains 25K positive and negative movie reviews each. First, let's install the bert-for-tf2 package. 

# In[ ]:


get_ipython().system('pip install bert-for-tf2')


# In[5]:


import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer


# In[3]:


import os
import re
import sys
import math
import datetime
import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow import keras


# In[4]:


print ('Tensorflow: ', tf.__version__)
print ('Python: ', sys.version)


# ### 2.1 Download Movie Review Data and BERT Weights
# 
# Here, we will download, extracts and import the IMDB large movie review dataset.

# In[ ]:


from tensorflow import keras 
import os 
import re

# load all files from the directory into a dataframe
def load_directory_data(directory):
    data ={}
    data['sentence'] = []
    data['sentiment'] = []
    for file_path in os.listdir(directory): 
        with tf.io.gfile.GFile(os.path.join(directory, file_path), 'r') as f: 
    data['sentence'].append(f.read())
    data['sentiment'].append(re.match('\d+_(\d+)\.txt',file_path).group(1))
    return pd.DataFrame.from_dict(data)

# combine positive and negative reviews into a dataframe; add a polarity column 
def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, 'pos'))
    neg_df = load_directory_data(os.path.join(directory,'neg'))
    pos_df['polarity'] = 1
    neg_df['polarity'] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

# download dataset from link
def download_and_load_datasets(force_download=False):
    dataset = tf.keras.utils.get_file(
        fname = 'acImbd.tar.gz',
        origin = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
        extract = True)
  
    train_df = load_dataset(os.path.join(os.path.dirname(dataset),"aclImdb",'train'))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset),"aclImdb",'test'))

    return train_df, test_df


# Let's use the MovieReviewData class below, to prepare/encode the data for feeding into our BERT model, by:
# - tokenizing the text
# - trim or pad it to a max_seq_len length
# - append the special tokens [CLS] and [SEP]
# - convert the string tokens to numerical IDs using the original model's token encoding from vocab.txt

# In[ ]:


class MovieReviewData:
    DATA_COLUMN = 'sentence'
    LABEL_COLUMN = 'polarity'
    
    def __init__(self, tokenizer: FullTokenizer, sample_size =None, max_seq_len =1024):
        self.tokenizer = tokenizer
        self.sample_size = sample_size
        self.max_seq_len = 0
        train, test = download_and_load_datasets()

        train,test =map(lambda df: df.reindex(df[MovieReviewData.DATA_COLUMN].str.len().sort_values().index),[train,test])

        if sample_size is not None:
            train, test = train.head(sample_size), test.head(sample_size)
    
        ((self.train_x, self.train_y),
        (self.test_x, self.test_y)) = map(self._prepare, [train,test])

        print('max_seq_len', self.max_seq_len)
        self.max_seq_len = min(self.max_seq_len,max_seq_len)
        ((self.train_x, self.train_x_token_types),
        (self.test_x, self.test_x_token_types)) = map(self._pad,[self.train_x , self.test_x])
      
    def _prepare(self,df):
        x,y =[],[]
        with tqdm(total =df.shape[0], unit_scale=True) as pbar:
            for ndx, row in df.iterrows():
                text, label = row[MovieReviewData.DATA_COLUMN], row[MovieReviewData.LABEL_COLUMN]
                tokens = self.tokenizer.tokenize(text)
                tokens = ['[CLS]'] + tokens + ['[SEP]']
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                self.max_seq_len = max(self.max_seq_len , len(token_ids))
                x.append(token_ids)
                y.append(int(label))
                pbar.update()

    return np.array(x),np.array(y)

    def _pad(self,ids):
        x,t = [],[]
        token_type_ids = [0] * self.max_seq_len
        for input_ids in ids:
            input_ids = input_ids[:min(len(input_ids),self.max_seq_len - 2)]
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
            x.append(np.array(input_ids))
            t.append(token_type_ids)
        return np.array(x), np.array(t)
        


# This tutorial uses the pre-trained BERT model -- BERT-Base,Uncased, which could be downloaded here. https://github.com/google-research/bert. Users could save the BERT weights in the S3 bucket created early or use the local SageMaker enviroment and save the weights locally.

# In[ ]:


assert_path = 'asset'
bert_model_name = 'uncased_L-12_H-768_A-12'
bert_ckpt_dir = os.path.join(assert_path , bert_model_name)
bert_ckpt_file = os.path.join(bert_ckpt_dir, 'bert_model.ckpt')
bert_config_file = os.path.join(bert_ckpt_dir, 'bert_config.json')


# After getting both the IMDB movie review data and the BERT weights ready, we will use the S3 bucket to store these data by directly uploading them.
# 
# ![Screen%20Shot%202020-06-26%20at%207.08.21%20PM.png](attachment:Screen%20Shot%202020-06-26%20at%207.08.21%20PM.png)

# ### 2.2 Data Preprocessing with Adapter BERT
# 
# In this step, we are ready to fetch the data using the BERT tokenizer. We will take the first 128 tokens by setting the max_seq_len = 128 and a sample size of 2500 each for train and test data due to transformer memory concerns.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntokenizer = FullTokenizer(vocab_file =os.path.join(bert_ckpt_dir, 'vocab.txt'))\ndata = MovieReviewData(tokenizer, sample_size = 10*128*2, max_seq_len =128) # sample_size:5000")


# In[ ]:


print('train_x',data.train_x.shape)
print('train_x_token_types',data.train_x_token_types.shape)
print('train_y',data.train_y.shape)
print('test_x',data.test_x.shape)
print('max_seq_len',data.max_seq_len)


# In this tutorial, we will also be using Adapter BERT, which requires us to frezee the original BERT layers first. In short, adapter BERT is a more parameter efficient way for fine-tuning. Instead of using an entire new model for every task, adapter BERT adds only a few trainable parameters per task while achieving near state-of-the-art performance. For more information on adapter BERT, visit here: https://arxiv.org/abs/1902.00751

# In[ ]:


def flatten_layers(root_layer):
    if isinstance(root_layer, keras.layers.Layer):
        yield root_layer
    for layer in root_layer._layers:
        for sub_layer in flatten_layers(layer):
            yield sub_layer


def freeze_bert_layers(l_bert):
    """
    Freezes all but LayerNorm and adapter layers - see arXiv:1902.00751.
    """
    for layer in flatten_layers(l_bert):
        if layer.name in ["LayerNorm", "adapter-down", "adapter-up"]:
            layer.trainable = True
        elif len(layer._layers) == 0:
            layer.trainable = False
        l_bert.embeddings_layer.trainable = False


def create_learning_rate_scheduler(max_learn_rate=5e-5,
                                   end_learn_rate=1e-7,
                                   warmup_epoch_count=10,
                                   total_epoch_count=90):

    def lr_scheduler(epoch):
        if epoch < warmup_epoch_count:
            res = (max_learn_rate/warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate*math.exp(
                math.log(end_learn_rate/max_learn_rate)*(epoch-warmup_epoch_count+1)/(total_epoch_count-warmup_epoch_count+1))
        return float(res)
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    return learning_rate_scheduler


# ### 2.3 Model Building and Training
# 
# Now, we are ready to create and train our model

# In[ ]:


def create_model(max_seq_len, adapter_size =64):
    
    #create the bert layer
    with tf.io.gfile.GFile(bert_config_file, 'r') as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = adapter_size
        bert = BertModelLayer.from_params(bert_params, name = 'bert')

    input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name = 'input_ids')
    output = bert(input_ids)

    print('bert shape',output.shape)
    cls_out = keras.layers.Lambda(lambda seq: seq[:,0,:])(output)
    cls_out = keras.layers.Dropout(0.5)(cls_out)
    logits = keras.layers.Dense(units=768, activation='tanh')(cls_out)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=2, activation = 'softmax')(logits)

    model = keras.Model(inputs = input_ids, outputs = logits)
    model.build(input_shape = (None, max_seq_len))

    load_stock_weights(bert, bert_ckpt_file)

    #freeze weights if adapter-BERT is used
    if adapter_size is not None:
        freeze_bert_layers(bert)
    
    model.compile(optimizer = keras.optimizers.Adam(),
                loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics = [keras.metrics.SparseCategoricalAccuracy(name = 'acc')])
    model.summary()

    return model


# In[ ]:


adapter_size = None 
model = create_model(data.max_seq_len, adapter_size = adapter_size) 


# In[ ]:


get_ipython().run_cell_magic('time', '', 'import datetime\nlog_dir = ".log/movie_reviews/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")\ntensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)\n\ntotal_epoch_count = 20\ncallbacks = [\n             create_learning_rate_scheduler(max_learn_rate=1e-5,\n                                                    end_learn_rate=1e-7,\n                                                    warmup_epoch_count=20,\n                                                    total_epoch_count=total_epoch_count),\n            keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),\n            tensorboard_callback]\n\nmodel.fit(x=data.train_x, y=data.train_y,\n          validation_split = 0.1,\n          batch_size = 12,\n          shuffle=True,\n          epochs = total_epoch_count,\n          callbacks=callbacks\n)')


# In[ ]:


#save model weights
model.save_weights('./movie_review.h5', overwrite=True)


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n_, train_acc = model.evaluate(data.train_x, data.train_y)\n_, test_acc = model.evaluate(data.test_x, data.test_y)\n\nprint('train acc', train_acc)\nprint('test acc', test_acc)")


# ### 2.4 Model Evaluation
# 
# For evaluation, let's load the previously saved model weights into a new model instance

# In[ ]:


model = create_model(data.max_seq_len, adapter_size=None)
model.load_weights('./movie_review.h5')


# Our model achieved a 91.5% accuracy on test data

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n_, test_acc = model.evaluate(data.test_x, data.test_y)\n\nprint('test acc',test_acc)")


# ## 3 BentoML SageMaker API Endpoints Deployment
# 
# In this section, we will demonstrate on using BentoML to build production-ready API endpoints and deploy it to AWS SageMaker. The core steps are as follow:
# 
# 1. Create a BentoML service file for model prediction <br>
# 2. Create and save a BentoMl packaged model called a BentoService bundle for model deployment<br>
# 3. Upload the BentoService bundle to cloud storage like S3 (optional)<br>
# 4. Use Bento CLI and its web UI for local testing<br> 
# 5. Deploy AWS SageMaker API endpoints through Bento CLI<br>
# 6. Use AWS CLI for endpoints testing<br>
# 
# Note: for AWS SageMaker deployment, you will need the following prerequisites: 1) Install and configure the AWS CLI
#  2) Install Docker 
# 
# for more information, please click here: https://docs.bentoml.org/en/latest/deployment/aws_sagemaker.html

# In[ ]:





# ### 3.1 Create a BentoML Service File for Prediction
# 
# First, let's create a prediction service file using BentoML. The three main BentoML concepts are: 
# 
# 1. Define the bentoml service environment 
# 2. Define the model artifacts based on the ML frameworks used for the trained model
# 3. Choose the relevant input adapters (formerly handlers) for the API
# 
# Note: BentoML supports a variety of major ML frameworks and input data format. For more details, please check available model artifacts here  
# https://docs.bentoml.org/en/latest/api/artifacts.html and adapters here https://docs.bentoml.org/en/latest/api/adapters.html
# 
# For defining the BentoML service environment and trouble-shooting, you would also use `auto_pip_dependencies= True` or pass the BentoML generated requirement.txt through `@bentoml.env(requirements_tex_file ='./requirements.txt')`

# In[19]:


get_ipython().run_cell_magic('writefile', 'bentoml_service.py', '\nimport tensorflow as tf\nimport numpy as np\nimport pandas as pd\n\nimport bentoml\nfrom bentoml.artifact import (TensorflowSavedModelArtifact, PickleArtifact)\nfrom bentoml.adapters import DataframeInput\n\nCLASSES  = [\'negative\',\'positive\']\nmax_seq_len = 128\n\ntry:\n    tf.config.set_visible_devices([],\'GPU\') \nexcept:\n    pass\n\n#define bentoml service environment\n@bentoml.env(pip_dependencies=[\'tensorflow\',\'bert\',\'bert-for-tf2\',\'numpy==1.18.1\',\'pandas==1.0.1\'])\n#define model artifacts\n@bentoml.artifacts([TensorflowSavedModelArtifact(\'model\'), PickleArtifact(\'tokenizer\')])\n\nclass Service(bentoml.BentoService):\n\n    def tokenize(self, inputs: pd.DataFrame):\n        tokenizer = self.artifacts.tokenizer\n        if isinstance(inputs, pd.DataFrame):\n            inputs = inputs.to_numpy()[:, 0].tolist()\n        else: \n            inputs = inputs.tolist()\n        pred_tokens = map(tokenizer.tokenize, inputs)\n        pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)\n        pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))\n        pred_token_ids = map(lambda tids: tids + [0] * (max_seq_len - len(tids)), pred_token_ids)\n        pred_token_ids = tf.constant(list(pred_token_ids), dtype=tf.int32)\n        return pred_token_ids\n    \n    # choose dataframe input adapter \n    @bentoml.api(input = DataframeInput(), md_max_latency = 300, mb_max_batch_size=20)\n    def predict(self, inputs):\n        model = self.artifacts.model\n        pred_token_ids = self.tokenize(inputs)\n        res = model(pred_token_ids).numpy().argmax(axis =-1)\n        return [CLASSES[i] for i in res]')


# ### 3.2 Create and Save BentoService Bundle
# 
# The following few lines of codes demonstrate the simplicity and time-saving benefits of using BentoML. Here, we first create a BentoService instance and then use the BentoService **pack method** to bundle our trained movie review model together. Finally, we use the BentoService **save method** to save this BentoService bundle, which is now ready for inference. This process eliminates the needs for reproducing the same prediction service for testing and production environment - making it easier for data science teams to deploy their models.
# 
# By default, the BentoService bundle is saved under `~/bentoml/repository/`directory. Users could also modify the model repository through BentoML's standalone component `YataiService`, for more information, please visit here: https://docs.bentoml.org/en/latest/concepts.html#model-management 
# 

# In[11]:


from bentoml_service import Service

#create a service instance for the movie review model
bento_svc = Service()

# pack model artifacts
bento_svc.pack('model',model)
bento_svc.pack('tokenizer',tokenizer)

#save the prediction service for model serving 
saved_path = bento_svc.save()


# ### Upload BentoService Bundle to S3 
# 
# As mentioned earlier, BentoML also provides ways to change the model repository - allowing data science teams to share the BentoService bundle easily for better collaborations. One way is by uploading it to the cloud services such as AWS S3. Using the same scripts as above and passing the S3 bucket URL into `.save()`, it will deploy the BentoService bundle directly into the S3 movie-review-dataset bucket we created earlier.

# In[ ]:


from bentoml_service import Service

#create a service instance for the movie review model
bento_svc = Service()

# pack model artifacts
bento_svc.pack('model',model)
bento_svc.pack('tokenizer',tokenizer)

#save the prediction service to aws S3
saved_path = bento_svc.save(''s3://movie-review-dataset/'')


# ![Screen%20Shot%202020-07-01%20at%2010.52.24%20PM.png](attachment:Screen%20Shot%202020-07-01%20at%2010.52.24%20PM.png)

# ### 3.3  Show Existing BentoServices
# 
# Using the BentoML CLI, we can see a list of BentoService generated here

# In[12]:


get_ipython().system('bentoml list')


# ### 3.4.1 Test REST API Locally -- Online API Serving
# 
# Before deploying the model to AWS SageMaker, we could test it locally first using the BentoML CLI. By using `bentoml serve`, it provides a near real-time prediction via API endpoints. 
# ![Screen%20Shot%202020-06-26%20at%201.45.31%20PM.png](attachment:Screen%20Shot%202020-06-26%20at%201.45.31%20PM.png)

# In[1]:


get_ipython().system('bentoml serve Service:20200702134432_033DAB  ')


# ![Screen%20Shot%202020-06-26%20at%201.45.14%20PM.png](attachment:Screen%20Shot%202020-06-26%20at%201.45.14%20PM.png)

# ### 3.4.2 Test REST API Locally -- Offline Batch Serving
# 
# Alternatively, we could also use `bentoml run` for local testing. BentoML provides many other model serving methods, such as: adaptive micro-batching, edge serving,and programmatic access. Please visit here: https://docs.bentoml.org/en/latest/concepts.html#model-serving 

# In[14]:


get_ipython().system('bentoml run Service:20200702134432_033DAB   predict --input \'["the acting was a bit lacking."]\'')


# ### 3.5 Deploy to AWS SageMaker
# 
# Finally, we are ready to deploy our BentoML packaged model to AWS SageMaker. We need to pass the deployment name, the BentoService name and the API name. Depending on the size of the BentoService generated, the deployment for this tutorial took about 30mins.

# In[16]:


get_ipython().system('bentoml sagemaker deploy sagemaker-moviereview-deployment -b Service:20200702134432_033DAB  --api-name predict')


# ### 3.6 Check Endpoint Service Status
# 
# Here we could confirm that the API endpoints status is InService 

# In[17]:


get_ipython().system('aws sagemaker describe-endpoint --endpoint-name dev-sagemaker-moviereview-deployment')


# ### 3.7 Test API Endpoints Using Boto3 SDK
# 
# Now, we are ready to test the SageMaker API endpoints by creating a small script using the boto3 SDK. Alternatively, users could also use the AWS CLI to test the endpoint. Please visit https://awscli.amazonaws.com/v2/documentation/api/latest/reference/sagemaker-runtime/invoke-endpoint.html

# In[1]:


import boto3
import json

endpoint = 'dev-sagemaker-moviereview-deployment'
runtime = boto3.Session().client('sagemaker-runtime')

movie_example = '["The acting was a bit lacking."]'

response = runtime.invoke_endpoint(EndpointName=endpoint, ContentType='application/json', Body=movie_example)
# Unpack response
result = json.loads(response['Body'].read().decode())

print(result)


# ## 4 Terminate AWS Resources 
# 
# Lastly, do not forget to terminate the AWS resources used in this tutorial. Users could also clean up used resources by logging into the SageMaker console. For more information, please see here:
# https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-cleanup.html

# In[4]:


bucket_to_delete = boto3.resource('s3').Bucket('movie-review-dataset')
bucket_to_delete.objects.all().delete()


# In[3]:


sagemaker.Session().delete_endpoint('dev-sagemaker-moviereview-deployment')


# In[ ]:





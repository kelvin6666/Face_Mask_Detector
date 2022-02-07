* THIS PROJECT IS RECOMMENDED TO RUN ON ANACONDA PROMPT
* MAKE SURE THAT YOU HAVE CUDA 11.1 + CUDNN INSTALLED

-----------Download Zip file--------------
1) Download the zip file needed via this link
https://drive.google.com/drive/folders/1Q3QqsUfVRYHxhaPVWvp92ttyOgmubgxM?usp=sharing

-----------Create Environment for Yolov5--------------
1) conda create --name v5
2) activate v5
3) pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
4) cd face_mask/yolov5/yolov5
5) pip install -r requirements.txt
6) nvcc --version (make sure that your CUDA is 11.1)

-----------To Train The Model On Our Dataset--------------

These are the parameters that you can change accordingly to you computer specifications
Model			Img Sizes		Batch size		Workers				Name
- yolov5l		- any image sizes	- any batch sizes	- It should be lower thant	- Give a name to your training
- yolov5l6		  that you preffered	  that you preffered	  the number of the cores of
- yolov5m								  your CPU.
- yolov5m6
- yolov5m_Objects365
- yolov5n
- yolov5n6
- yolov5s
- yolov5s6
- yolov5x
- yolov5x6

Example Code to initiate the training:
python train.py --img 320 --cfg yolov5m.pt --data mask.yaml --batch-size 8 --epochs 100 --workers 3 --name VIP_8_320_yolov5m

-----------To use our Yolov5 model--------------
We have train 15 yolov5 models:
- yolov5m_320_4
- yolov5m_320_8
- yolov5m_320_16
- yolov5m_320_32
- yolov5m_640_4
- yolov5m_640_8
- yolov5m_640_16
- yolov5m_640_32
- yolov5s_320_4
- yolov5s_320_8
- yolov5s_320_16
- yolov5s_320_32
- yolov5s_640_8
- yolov5s_640_16

and yolov5s_640_32 which is the best model
The code below can be used to demo our best model in real time:
python detect.py --weights "weight/best.pt" --source 0

To run any other model just replace best with the model name stated above. For example to use yolov5m_320_4:
python detect.py --weights "weight/yolov5m_320_4.pt" --source 0

--------------------------------------------- To run SSD, Faster R-CNN, and EfficientDet ---------------------------------------------

-----------Create Environment for Tensorflow Object Detection API--------------
1) Unzip the file and navigate into Tensorflow Object Detection API Submission
2) conda create --name tflow_env
3) activate tflow_env
4) pip install -r requirements.txt
5) open the jupyter notebook and follow the instructions inside

-----------Explanation of instructions in jupyter notebook--------------
1) There are 12 sections in each jupyter notebook
2) Run step 1, 2, and 3 to install object detection packages into virtual env,
   this step takes some times and will always fail at the beginning, need to try
   again and again 
3) Once the 3rd step verification shows "OK" at the end of running, this mean you
   successfully install the packages and ready to proceed to next step, else, you
   need to look and the error and reinstall the packages needed.Every time we install 
   it at new virtual environment, different error are shown, so we hard to determine
   which error will occurs at your site. But based on our experience installing at 4 pc,
   the errors are just "xxx package not found", just install that package, re-run again 
   and should be ok.
4) After step 3, if want to train the model, straight away go to step 9. The code below can 
   used to train the model :
   python path_to_model_main_tf2.py --model_dir=path_to_model --pipeline_config=path_to_pipeline.config --num_train_steps=total_training_step

   Example : 
   python Tensorflow\models\research\object_detection\model_main_tf2.py --model_dir=Tensorflow\workspace\models\my_ssd_mobnet_small_4bs --pipeline_config_path=Tensorflow\workspace\models\my_ssd_mobnet_small_4bs\pipeline.config 
   --num_train_steps=72500

   Note that: Step 4,5,6,7,8 is not require as we already configure the files needed.

5) If want to evaluate the model, straight away go to step 10. We have trained 14 models:
   A)List of model's name that we have trained:
   -efficientdet_d0
   -efficientdet_d0_bs4
   -efficientdet_d0_bs16
   -faster_rcnn_resnet50
   -faster_rcnn_resnet101
   -faster_rcnn_resnet152
   -my_ssd_mobnet_small
   -my_ssd_mobnet_small_4bs
   -my_ssd_mobnet_small_8bs
   -my_ssd_mobnet_small_32bs
   -my_ssd_resnet50
   -my_ssd_resnet50_bs4
   -my_ssd_resnet50_bs8
   -my_ssd_resnet50_bs32

   The code below can used to evaluate the model's performance:
   python path_to_model_main_tf2.py --model_dir=path_to_model --pipeline_config=path_to_pipeline.config --checkpoint_dir=path_to_checkpoint_dir

   Example:
   python Tensorflow\models\research\object_detection\model_main_tf2.py --model_dir=Tensorflow\workspace\models\my_ssd_mobnet_small_4bs --pipeline_config_path=Tensorflow\workspace\models\my_ssd_mobnet_small_4bs\pipeline.config --checkpoint_dir=Tensorflow\workspace\models\my_ssd_mobnet_small_4bs

   Note : Just change the model's name for evaluating different models

6) If want to detect the image, run step 11 and 12 in the jupyter notebook.



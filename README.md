# Official Implementation of DDC3N: Doppler-Driven Convolutional 3D Network for Human Action Recognition
 This is for our ongoing project on action recognition of sports athletes.
 
![image](https://github.com/muxiddin19/DDC3N--Doppler-Driven-Convolutional-3D-Network-for-Human-Action-Recognition/assets/54941476/d6b447f2-0b35-41fc-877a-93dc3203fb52)

# Abstract

In deep learning (DL)-based human action recognition (HAR), considerable strides have been undertaken. Nevertheless, the precise classification of sports athletes' actions still needs to be completed. Primarily attributable to the exigency for exhaustive datasets about sports athletes' actions and the enduring quandaries imposed by variable camera perspectives, mercurial lighting conditions, and occlusions. This investigative endeavor thoroughly examines extant HAR datasets, thereby furnishing a yardstick for gauging the efficacy of cutting-edge methodologies. In light of the paucity of accessible datasets delineating athlete actions, we have taken a proactive stance, endeavoring to curate two meticulously tailored datasets tailored explicitly for sports athletes, subsequently scrutinizing their consequential impact on performance enhancement.
While the superiority of 3D convolutional neural networks (3DCNN) over graph convolutional networks (GCN) in HAR is evident, it must be acknowledged that they entail a considerable computational overhead, particularly when confronted with voluminous datasets. Our inquiry introduces innovative methodologies and a more resource-efficient remedy for HAR, thereby alleviating the computational strain on the 3DCNN architecture. Consequently, it proffers a multifaceted approach towards augmenting HAR within the purview of surveillance cameras, bridging lacunae, surmounting computational impediments, and effectuating significant strides in the accuracy and efficacy of HAR frameworks.

# Dependencies
- python >= 3.6
- decord >= 0.4.1
- einops
- matplotlib
- numpy
- opencv-contrib-python
- Pillow
- scipy
- torch>=1.3
- coverage
- flake8
- interrogate
- isort==4.3.21
- pytest
- pytest-runner
- xdoctest >= 0.10.0
- yapf

# Data Preparation 
## Data examples
### Video Data
The images for each class were collected and compressed in the Google Drive below, and you can download them from the folder below. 
An example of a https://drive.google.com/drive/folders/127-hhaBebuUMqA38OxyMtLCrCzLjyYNA?usp=sharing skeleton rendered video is shown in the link https://drive.google.com/file/d/1wjaW8oeZeVYRyd-34HXQpJPhIukIa5Pk/view?usp=sharing .

### Skeleton Data
We uploaded 20220905, 20220914 folders as an example, you can download and use the during the experiment, while whole data is not available due to the completing the uploading process. https://drive.google.com/drive/folders/16N--bblvs2vtoU4tcqduxrEk9aTA2WK1?usp=share_link

## Only Pose data

This is the code that is post-processed to put the result from pose extraction mentioned above into the model. The result is the splitted data into three parts, training, testing and validation sets, that is ready for the experiments. Where 20220705, 20220706, ... are folders containing the related pose data, that is in the form of a simple json file that does not contain metadata.

python create_data.py --folders 20220705 20220706 20220707 20220708 20220711 --trainset True --full True 

You should check all the details inside the code, as location of data might be different due to exploited OS or folders name.

- Trainset True is a code that creates a csv and also creates a training set, so it has to be True to make a train test set,
- Full True is True: Make all normal error data.
- False: make only normal It's designed to be written like this.

## Cutting the video data through the given metada.
To increase performance of the HAR model, we exluded frames from the given data, where pose extarction results have low performance. Hence we have metada that has starting_frame, and end_frame values for each data, whre pose extraction model has high performance. After performing the related operation, we have data with only high pose estimation performance.

- python cut_data_through_the given_metadata.py --folders 20220705 20220706 20220707 20220708 20220711 --trainset True --full True 
It is not mandatory, but help to increase the relted performance of the last model.

## Splitted Pose Dataset, pretrained weights, and the related config files for downloading:
If you train only with pose data you can download the related data
There are total four folders for CrossFit normal, CrossFit full, Figure Skating normal,  and each folder contains 4 files.

They are:
1. pth file for saved weights
2. config.py file
3. test pickle file 
4. validation pickle file
   
Train file was not uploaded due to its huge size, however, you can use one of two uploaded file as a training, and split another one for testing and validation set, depending on your experiment setting.

https://drive.google.com/file/d/11Hp2n4K-u1674ddxvomFlP-0AcEmpwRl/view?usp=share_link

## Dopler driven block

### Extracting CWSTB features
- The related codes will be uploaded soon, while modification process is done.

### Extracting Doppler features
- The related codes will be uploaded soon, while modification process is done.

## Concatenation of the obtained features
- The related codes will be uploaded soon, while modification process is done.

## Visualization of the final data

![image](https://github.com/muxiddin19/DDC3N--Doppler-Driven-Convolutional-3D-Network-for-Human-Action-Recognition/assets/54941476/cf84b2ca-5150-4144-948c-ea034f735042)

# Training Process
- python tools/train.py pth/figure_normal/figure_normal_config.py --work-dir work_dirs/figure_normal --validate --test-best --gpus 2 --seed 0 --deterministic
- for the further explanation of the code, please refer to train.py file which provides the related info in detailes
  
## Resume Interrupted training
- python tools/train.py configs/skeleton/posec3d/doppler_CF_full.py --work-dir work_dirs/custom_dopp_FC_full --resume-from work-dirs/custom_doppler/latest.pth --validate --test-best --gpus 2 --seed 0 --deterministic

As traing requires huge of time, there might be cases whre training process is interrupted. It can be resumed from the last saved epoch weight as given in above code.

## Training from the pretrained weights

- python tools/train.py pth/figure_normal/figure_normal_config.py --resume-from pth/figure_normal/figure_normal.pth --work-dir work_dirs/figure_normal_from_pretrained --validate --test-best --gpus 2 --seed 0 --deterministic

   While pretrained weight is given it can used, to save time, human and computational expenses.

  ## Testing Process

- python tools/test.py pth/figure_normal/figure_normal_config.py --work-dir work_dirs/figure_normal/last.pth --eval top_k_accuracy mean_class_accuracy --out result_fig_norm_last.pkl

Testing code differs from the training one with the weights, and evaluation metrics, while both use the same config file.

  ## Citation

- The related citation will be updated soon.

  ## Acknowledgement

This repo is actively exploited https://github.com/open-mmlab/mmaction2, https://github.com/bruceyo/MMNet, https://github.com/open-mmlab/mmaction2/tree/main/configs/skeleton/posec3d, https://github.com/Oli21-chen/STM-spatiotemporal-and-motion-encoding, https://github.com/Jho-Yonsei/HD-GCN, https://github.com/stnoah1/infogcn, https://github.com/Uason-Chen/CTR-GCN/tree/main and other related repos. 

Thanks to the original authors for their awesome works!

  ## Contact

  For any questions, feel free to contact: trinity@inha.ac.kr, muhiddin1979@inha.edu.

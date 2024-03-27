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

# ALADDIN-BiGAN-anomaly-detection

This directory contains the files used for the development of "[**_Unsupervised Anomaly Detection for Underwater Gliders Using Generative Adversarial Networks_**](https://doi.org/10.1016/j.engappai.2021.104379)".
* images: anomaly detection using BiGAN for underwater gliders: (a) training using normal data and (b) testing using unseen deployment data,
* results: the anomaly detection results of the test deployments presented in the paper,
* saved_model: pre-trained neural networks,
* anomaly_multi_sensitivity.py: main script to train and test the anomaly detection systems, including a sensitivity study detailed in the paper,
* data_processing_sensitivity.py: processes the raw datasets, generating datasets for anomaly detection system training, validation, test and sensitivity study,
* model.py: builds the neural networks.
* utilities.py: auxiliary functions for training and testing


<img src=/images/gan_anomaly.png width="500" title="Anomaly detection using BiGAN for underwater gliders:  (a) training using normal data and (b)testing using unseen deployment data.">

The datasets are mostly collected from [BODC's Glider inventory](https://www.bodc.ac.uk/data/bodc_database/gliders/).

Please cite the paper as below in any resulting publications.
```
@article{wu2021anomaly,
  title={Unsupervised Anomaly Detection for Underwater Gliders Using Generative Adversarial Networks},
  author={Wu, P. and Harris, C.A. and Salavasidis, G. and Lorenzo-Lopez, A. and Kamarudzaman, I. and Philips, A.B. and Thomas, G. and Anderlini, E.},
  journal={Engineering Applications of Artificial Intelligence},
  volume={104},
  pages={104379},
  year={2021},
  publisher={Elsevier}
}
```

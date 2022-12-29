# Deep learning for tongue muscle fiber reconstruction

This folder contains the code and example scripts to train and test the deep network for tongue muscle fiber reconstruction from HARDI data.

## Folders and files description
 - Folder _dirdetect/_ contains tongue muscle fiber orientation reconstruction code.
   * File _preprocessing.py_ generates spherical harmonic coefficients from HARDI image.
   * File _dataset.py_ defines pytorch dataset for network.
   * File _net.py_ defines fiber reconstruction deep learning network structures.
   * File _loss.py_ defines the combined loss function, including the MSE loss in the first stage and BCE, angular error, and separation loss in the second stage.
   * File _trainer.py_ defines the trainer.
   * File _predictor.py_ defines the predictor to apply the network to testing data.
 - Folder _scripts/_ contains example scripts to train and test the model.
 - Trained model link: [Fiber reconstruction model](https://1drv.ms/u/s!AlZc6vjzuawH-0OHKzb4T-tF1Pf-?e=IzNsfR)

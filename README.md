
## Official code for _Weighted Point Cloud Normal Estimation_ in ICME 2023

Experiments are run on:

```
Ubuntu 16.04
CUDA 11.0
python 3.7
pytorch 1.7.0
torchvision 0.8.0
numpy 1.21.2
scikit-learn 1.0.2
scipy 1.7.3
matplotlib 3.5.1
tqdm 4.64.0
plyfile 0.7.4
```

Please make sure you have installed the required packages before running the code.

To run our code, first go to this directory:
```
cd data/pclouds
```

Then run download_pcpnet_data.py to download PCPNet dataset:
```
python download_pcpnet_data.py
```

The script will download and unzip the data automatically. After the unzipping process finishes, go to this project's root directory:
```
cd ../..
```


After that, you can test the synthetic shapes using our pre-trained models by running:
```
python test_pcpnet.py --saved_path pretrained_models
```
The results will appear in this folder:
```
./data/Result
```
The output shapes are in this format: shapename_noiselevel_pred.xyz (e.g., galera100k_noise_white_3.00e-02_pred.xyz). You can visualise each shape's normals using 3D software such as MeshLab.



To evaluate, run:
```
python run_evaluate_multi_noise_pidx.py
```

This evaluates the synthetic shapes based on PCPNet's pre-defined point indices (.pidx files). The RMSE and PGP results for each noise level will be summarised in the folder
```
./data/Result/summary_pidx/
```
For example, ```testset_036_noise_evaluation_results.txt``` shows results on 0.36% noise level. The RMSE is listed in "RMS not oriented (shape average)".


Finally, if you wish to train the model, please run:

```
python run_training.py
```





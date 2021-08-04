# Generate PI with NN
This repository contains the code I developed during my internship at DataShape.
The goal was to build a neural network that could predict a persistence image from a point cloud.

The repository is divided in several folders and files, the detail is listed below.

 List of the different notebooks
---

- **Training of the NN**

   This folder contains the notebooks used for the training of the NN that generates PI.
   1.  `Training of the NN for dynamical systems` is the definition and the training of the NN that predicts PI for the dynamical systems.
   2. ` Training of the NN for multiple circles` is the definition and the training of the NN that predicts PI for the multiple circle dataset.
   3. `Training of the NN for one smaller circle` is the definition and the training of the NN that predicts PI for the dataset with only one circle.
   4. `Training of the NN for the torus` is the definition and the training of the NN that predicts PI for the dataset with the torus.
   5. `Training of the NN for embedded data in higher dimension` is the definition and the training of the NN that predicts PI for the embedded in higher dimension dataset. (4d using the Whitney embedding and 10d using a matrix multiplication).

- **Study of the results of the NN**

  This folder contains the notebooks used to study the NN trained on different dataset, by doing some statistical tests or classification and regression.
  1. `MMD test on all datasets` is the notebook used to compute the MMD test on each NN and to have the p-value of this test.
  2. `Study of the results of the NN for dynamical system` is the notebook used to compute the classification results and the KS test for the NN trained on dynamical systems.
  3. `Study of the results of the NN for multiple circles`is the notebook used to compute the classification results and the KS test for the NN trained on the multiple circles dataset.
  4. `Study of the results of the NN for multiple noisy circles` is the notebook used to compute the classification results and the KS test for the NN trained on the multiple noisy circles dataset using the DTM filtration.
  5. `Study of the results of the NN for one noisy circle` is the notebook used to compute the regression results and the KS test for the NN trained on the one  noisy circle dataset.
  6. `Study of the results of the NN for one smaller circle` is the notebook used to compute the regression results and the KS test for the NN trained on the one smaller circle dataset.
  7. `Study of the results of the NN for the sphere` is the notebook used to compute the regression results and the KS test for the NN trained on the one sphere dataset.
  8. `Study of the results of the NN for the torus` is the notebook used to compute the regression results and the KS test for the NN trained on the one torus dataset.

# Offensive AI Compilation

A curated list of useful resources that cover Offensive AI.

## Abuse

Exploiting the vulnerabilities of AI models. 

### Adversarial Machine Learning

Adversarial Machine Learning is responsible for assessing their weaknesses and providing countermeasures.

#### Attacks

It is organized in four types of attacks: extraction, inversion, poisoning and evasion.

![Adversarial Machine Learning attacks](/assets/attacks.png)

##### Extraction

It tries to steal the parameters and hyperparameters of a model by making requests that maximize the extraction of information.

![Extraction attack](/assets/extraction.png)

Depending on the knowledge of the adversary's model, white-box and black-box attacks can be performed.

In the simplest white-box case (when the adversary has full knowledge of the model, e.g., a sigmoid function), one can create a system of linear equations that can be easily solved.

In the generic case, where there is insufficient knowledge of the model, the substitute model is used. This model is trained with the requests made to the original model in order to imitate the same functionality as the original one.

![White-box and black-box extraction attacks](/assets/extraction-white-black-box.png)

###### Limitations

  * Training a substitute model is equivalent (in many cases) to training a model from scratch.

  * Very computationally intensive.

  * The adversary has limitations on the number of requests before being detected.

###### Defensive actions

  * Rounding of output values.

  * Use of [differential privacy](https://en.wikipedia.org/wiki/Differential_privacy).

  * Use of ensembles.

  * Use of specific defenses
    * [Specific architectures](https://arxiv.org/abs/1711.07221)
    * [PRADA](https://arxiv.org/abs/1805.02628)
    * [Adaptive Misinformation](https://arxiv.org/abs/1911.07100)
    * ...

##### Inversion (or inference)
  
They are intended to reverse the information flow of a machine learning model.

![Inference attack](/assets/inference.png)

They enable an adversary to have knowledge of the model that was not explicitly intended to be shared.

They allow to know the training data or information as statistical properties of the model.

Three types are possible:

  * **Membership Inference Attack (MIA)**: An adversary attempts to determine whether a sample was employed as part of the training. 

  * **Property Inference Attack (PIA)**: An adversary aims to extract statistical properties that were not explicitly encoded as features during the training phase.

  * **Reconstruction**: An adversary tries to reconstruct one or more samples from the training set and/or their corresponding labels. Also called inversion.


###### Defensive actions

  * Use of advanced cryptography. Countermeasures include differential privacy, homomorphic cryptography and secure multiparty computation.

  * Use of regularization techniques such as Dropout due to the relationship between overtraining and privacy.

  * Model compression has been proposed as a defense against reconstruction attacks.

##### Poisoning

They aim to corrupt the training set by causing a machine learning model to reduce its accuracy.

![Poisoning attack](/assets/poisoning.png)

This attack is difficult to detect when performed on the training data, since the attack can propagate among different models using the same training data.

The adversary seeks to destroy the availability of the model by modifying the decision boundary and, as a result, producing incorrect predictions or, create a backdoor in a model. In the latter,  the model behaves correctly (returning the desired predictions) in most cases, except for certain inputs specially created by the adversary that produce undesired results. The adversary can manipulate the results of the predictions and launch future attacks.

###### Backdoors

[BadNets](https://arxiv.org/abs/1708.06733) are the simplest type of backdoor in a machine learning model. Moreover, BadNets are able to be preserved in a model, even if they are retrained again for a different task than the original model (transfer learning).

It is important to note that **public pre-trained models may contain backdoors**.

###### Defensive actions

  * Detection of poisoned data, along with the use of data sanitization.

  * Robust training methods.

  * Specific defenses.

##### Evasion

An adversary adds a small perturbation (in the form of noise) to the input of a machine learning model to make it classify incorrectly (example adversary).

![Evasion attack](/assets/evasion.png)

They are similar to poisoning attacks, but their main difference is that evasion attacks try to exploit weaknesses of the model in the inference phase. 

The goal of the adversary is for adversarial examples to be imperceptible to a human.

Two types of attack can be performed depending on the output desired by the opponent:

  * **Targeted**: the adversary aims to obtain a prediction of his choice.

    ![Targeted attack](/assets/targeted.png)

  * **Untargeted**: the adversary intends to achieve a misclassification.

    ![Untargeted attack](/assets/untargeted.png)

The most common attacks are **white-box attacks**:

  * L-BFGS
  * FGSM
  * BIM
  * JSMA
  * Carlini & Wagner (C&W)
  * NewtonFool
  * EAD
  * BIM
  * UAP
  * ...

###### Defensive actions

  * Adversarial training, which consists of crafting adversarial examples during training so as to allow the model to learn features of the adversarial examples, making the model more robust to this type of attack.

  * Transformations on inputs.

  * Gradient masking/regularization. Not very effective.

  * Weak defenses.

#### Tools 

| Name | Type | Supported algorithms | Supported attack types | Attack/Defence | Supported frameworks | Popularity |
| ---------- | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
| [Cleverhans](https://github.com/cleverhans-lab/cleverhans) | Image | Deep Learning | Evasion | Attack | Tensorflow, Keras, JAX | [![stars](https://badgen.net/github/stars/cleverhans-lab/cleverhans)](https://github.com/cleverhans-lab/cleverhans)|
| [Foolbox](https://github.com/bethgelab/foolbox) | Image | Deep Learning | Evasion | Attack | Tensorflow, PyTorch, JAX | [![stars](https://badgen.net/github/stars/bethgelab/foolbox)](https://github.com/bethgelab/foolbox)|
| [ART](https://github.com/Trusted-AI/adversarial-robustness-toolbox) | Any type (image, tabular data, audio,...) | Deep Learning, SVM, LR, etc. | Any (extraction, inference, poisoning, evasion) | Both | Tensorflow, Keras, Pytorch, Scikit Learn | [![stars](https://badgen.net/github/stars/Trusted-AI/adversarial-robustness-toolbox)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)|
| [TextAttack](https://github.com/QData/TextAttack) | Text | Deep Learning | Evasion | Attack | Keras, HuggingFace | [![stars](https://badgen.net/github/stars/QData/TextAttack)](https://github.com/QData/TextAttack)|
| [Advertorch](https://github.com/BorealisAI/advertorch) | Image | Deep Learning | Evasion | Both | --- | [![stars](https://badgen.net/github/stars/BorealisAI/advertorch)](https://github.com/BorealisAI/advertorch)|
| [AdvBox](https://github.com/advboxes/AdvBox) | Image | Deep Learning | Evasion | Both | PyTorch, Tensorflow, MxNet | [![stars](https://badgen.net/github/stars/advboxes/AdvBox)](https://github.com/advboxes/AdvBox)|
| [DeepRobust](https://github.com/DSE-MSU/DeepRobust) | Image, graph | Deep Learning | Evasion | Both | PyTorch | [![stars](https://badgen.net/github/stars/DSE-MSU/DeepRobust)](https://github.com/DSE-MSU/DeepRobust)|
| [Counterfit](https://github.com/Azure/counterfit) | Any | Any | Evasion | Attack | --- | [![stars](https://badgen.net/github/stars/Azure/counterfit)](https://github.com/Azure/counterfit)|
| [Adversarial Audio Examples](https://github.com/carlini/audio_adversarial_examples) | Audio | DeepSpeech | Evasion | Attack | --- | [![stars](https://badgen.net/github/stars/carlini/audio_adversarial_examples)](https://github.com/carlini/audio_adversarial_examples)|

###### ART

[Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox), abbreviated as ART, is an open-source Adversarial Machine Learning library for testing the robustness of machine learning models.

![ART logo](/assets/art.png)

It is developed in Python and implements extraction, inversion, poisoning and evasion attacks and defenses.

ART supports the most popular frameworks: Tensorflow, Keras, PyTorch, MxNet, ScikitLearn, among many others.

It is not limited to the use of models that use images as input, but also supports other types of data, such as audio, video, tabular data, etc.

###### Cleverhans

[Cleverhans](https://github.com/cleverhans-lab/cleverhans) is a library for performing evasion attacks and testing the robustness of a deep learning model on image models.

![Cleverhans logo](/assets/cleverhans.png)

It is developed in Python and integrates with the Tensorflow, Torch and JAX frameworks.

It implements numerous attacks such as L-BFGS, FGSM, JSMA, C&W, among others.

## Use

### Audio

#### Tools

#### Applications

#### Detection

### Image

#### Tools

#### Applications

#### Detection

### Video

#### Tools

#### Applications

#### Detection

### Text

#### Tools

#### Detection

#### Applications

## Contributors

<table>
  <tr>
    <td align="center"><a href="https://github.com/Miguel000"><img src="https://avatars2.githubusercontent.com/u/13256426?s=460&v=4" width="150;" alt=""/><br /><sub><b>Miguel Hernández</b></sub></a></td>
    <td align="center"><a href="https://github.com/jiep"><img src="https://avatars2.githubusercontent.com/u/414463?s=460&v=4" width="150px;" alt=""/><br /><sub><b>José Ignacio Escribano</b></sub></a></td>
  </tr>
</table>

## License

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

* [Creative Commons Attribution Share Alike 4.0 International](LICENSE.txt)
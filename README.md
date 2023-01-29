# Offensive AI Compilation

A curated list of useful resources that cover Offensive AI.

## Abuse

Exploiting the vulnerabilities of AI models. 

### Adversarial Machine Learning

Adversarial Machine Learning is responsible for assessing their weaknesses and providing countermeasures.

It is organized in four types of attacks: extraction, inversion, poisoning and evasion.

![Adversarial Machine Learning attacks](/assets/attacks.png)

##### Attacks

* Extraction: It tries to steal the parameters and hyperparameters of a model by making requests that maximize the extraction of information.

  ![Extraction attack](/assets/extraction.png)

  Depending on the knowledge of the adversary's model, white-box and black-box attacks can be performed.

  In the simplest white-box case (when the adversary has full knowledge of the model, e.g., a sigmoid function), one can create a system of linear equations that can be easily solved.

  In the generic case, where there is insufficient knowledge of the model, the substitute model is used. This model is trained with the requests made to the original model in order to imitate the same functionality as the original one.

  ![White-box and black-box extraction attacks](/assets/extraction-white-black-box.png)

* Inversion

* Poisoning

* Evasion

#### Tools 

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
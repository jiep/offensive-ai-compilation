# Offensive AI Compilation

A curated list of useful resources that cover Offensive AI.

## 📁 Contents 📁
- [🚫 Abuse 🚫](#-abuse-)
  - [🧠 Adversarial Machine Learning 🧠](#-adversarial-machine-learning-)
    - [⚡ Attacks ⚡](#-attacks-)
      - [🔒 Extraction 🔒](#-extraction-)
        - [⚠️ Limitations ⚠️](#️-limitations-️)
        - [🛡️ Defensive actions 🛡️](#️-defensive-actions-️)
        - [🔗 Useful links 🔗](#-useful-links-)
      - [⬅️ Inversion (or inference) ⬅️](#️-inversion-or-inference-️)
        - [🛡️ Defensive actions 🛡️](#️-defensive-actions-️-1)
        - [🔗 Useful links 🔗](#-useful-links--1)
      - [💉 Poisoning 💉](#-poisoning-)
        - [🔓 Backdoors 🔓](#-backdoors-)
        - [🛡️ Defensive actions 🛡️](#️-defensive-actions-️-2)
        - [🔗 Useful links 🔗](#-useful-links--2)
      - [🏃‍♂️ Evasion 🏃‍♂️](#️-evasion-️)
        - [🛡️ Defensive actions 🛡️](#️-defensive-actions-️-3)
        - [🔗 Useful links 🔗](#-useful-links--3)
    - [🛠️ Tools 🛠️](#️-tools-️)
        - [ART](#art)
        - [Cleverhans](#cleverhans)
- [🔧 Use 🔧](#-use-)
  - [🕵️‍♂️ Pentesting 🕵️‍♂️](#️️-pentesting-️️)
  - [🦠 Malware 🦠](#-malware-)
  - [🗺️ OSINT 🗺️](#️osint-️)
  - [📧 Phishing 📧](#phishing-)
  - [👨‍🎤 Generative AI 👨‍🎤](#-generative-ai-)
    - [🔊 Audio 🔊](#-audio-)
      - [🛠️ Tools 🛠️](#️-tools-️-1)
      - [💡 Applications 💡](#-applications-)
      - [🔎 Detection 🔎](#-detection-)
    - [📷 Image 📷](#-image-)
      - [🛠️ Tools 🛠️](#️-tools-️-2)
      - [💡 Applications 💡](#-applications--1)
      - [🔎 Detection 🔎](#-detection--1)
    - [🎥 Video 🎥](#-video-)
      - [🛠️ Tools 🛠️](#️-tools-️-3)
      - [💡 Applications 💡](#-applications--2)
      - [🔎 Detection 🔎](#-detection--2)
    - [📄 Text 📄](#-text-)
      - [🛠️ Tools 🛠️](#️-tools-️-4)
      - [🔎 Detection 🔎](#-detection--3)
      - [💡 Applications 💡](#-applications--3)
  - [📚 Misc 📚](#-misc-)
- [📊 Surveys 📊](#-surveys-)
- [🗣 Contributors 🗣](#-contributors-)
- [©️ License ©️](#️-license-️)

## 🚫 Abuse 🚫

Exploiting the vulnerabilities of AI models.

### 🧠 Adversarial Machine Learning 🧠

Adversarial Machine Learning is responsible for assessing their weaknesses and providing countermeasures.

#### ⚡ Attacks ⚡

It is organized into four types of attacks: extraction, inversion, poisoning and evasion.

![Adversarial Machine Learning attacks](/assets/images/attacks.png)

##### 🔒 Extraction 🔒

It tries to steal the parameters and hyperparameters of a model by making requests that maximize the extraction of information.

![Extraction attack](/assets/images/extraction.png)

Depending on the knowledge of the adversary's model, white-box and black-box attacks can be performed.

In the simplest white-box case (when the adversary has full knowledge of the model, e.g., a sigmoid function), one can create a system of linear equations that can be easily solved.

In the generic case, where there is insufficient knowledge of the model, the substitute model is used. This model is trained with the requests made to the original model in order to imitate the same functionality as the original one.

![White-box and black-box extraction attacks](/assets/images/extraction-white-black-box.png)

###### ⚠️ Limitations ⚠️

  * Training a substitute model is equivalent (in many cases) to training a model from scratch.

  * Very computationally intensive.

  * The adversary has limitations on the number of requests before being detected.

###### 🛡️ Defensive actions 🛡️

  * Rounding of output values.

  * Use of [differential privacy](https://en.wikipedia.org/wiki/Differential_privacy).

  * Use of [ensembles](https://en.wikipedia.org/wiki/Ensemble_learning).

  * Use of specific defenses
    * [Specific architectures](https://arxiv.org/abs/1711.07221)
    * [PRADA](https://arxiv.org/abs/1805.02628)
    * [Adaptive Misinformation](https://arxiv.org/abs/1911.07100)
    * ...

###### 🔗 Useful links 🔗

  * [Stealing Machine Learning Models via Prediction APIs](https://arxiv.org/abs/1609.02943)
  * [Stealing Hyperparameters in Machine Learning](https://arxiv.org/abs/1802.05351)
  * [Knockoff Nets: Stealing Functionality of Black-Box Models](https://arxiv.org/abs/1812.02766)
  * [Model Extraction Warning in MLaaS Paradigm](https://arxiv.org/abs/1711.07221)
  * [Copycat CNN: Stealing Knowledge by Persuading Confession with Random Non-Labeled Data](https://arxiv.org/abs/1806.05476)
  * [Prediction Poisoning: Towards Defenses Against DNN Model Stealing Attacks](https://arxiv.org/abs/1906.10908)
  * [Stealing Neural Networks via Timing Side Channels](https://arxiv.org/abs/1812.11720)
  * [Model Stealing Attacks Against Inductive Graph Neural Networks](https://arxiv.org/abs/2112.08331)
  * [High Accuracy and High Fidelity Extraction of Neural Networks](https://arxiv.org/abs/1909.01838)
  * [Poisoning Web-Scale Training Datasets is Practical](https://arxiv.org/abs/2302.10149)
  * [Polynomial Time Cryptanalytic Extraction of Neural Network Models](https://eprint.iacr.org/2023/1526)
  * [Prompt-Specific Poisoning Attacks on Text-to-Image Generative Models](https://arxiv.org/abs/2310.13828)
  * [Awesome Data Poisoning And Backdoor Attacks](https://github.com/penghui-yang/awesome-data-poisoning-and-backdoor-attacks): A curated list of papers & resources linked to data poisoning, backdoor attacks and defenses against them. [![stars](https://badgen.net/github/stars/penghui-yang/awesome-data-poisoning-and-backdoor-attacks)](https://github.com/penghui-yang/awesome-data-poisoning-and-backdoor-attacks)
  * [BackdoorBox](https://github.com/THUYimingLi/BackdoorBox): An Open-sourced Python Toolbox for Backdoor Attacks and Defenses. [![stars](https://badgen.net/github/stars/THUYimingLi/BackdoorBox)](https://github.com/THUYimingLi/BackdoorBox)
  * [Stealing Part of a Production Language Model](https://arxiv.org/abs/2403.06634)
  * [Hard-Label Cryptanalytic Extraction of Neural Network Models](https://eprint.iacr.org/2024/1403)

##### ⬅️ Inversion (or inference) ⬅️

They are intended to reverse the information flow of a machine-learning model.

![Inference attack](/assets/images/inversion.png)

They enable an adversary to know the model that was not explicitly intended to be shared.

They allow us to know the training data or information as statistical properties of the model.

Three types are possible:

  * **Membership Inference Attack (MIA)**: An adversary attempts to determine whether a sample was employed as part of the training.

  * **Property Inference Attack (PIA)**: An adversary aims to extract statistical properties that were not explicitly encoded as features during the training phase.

  * **Reconstruction**: An adversary tries to reconstruct one or more samples from the training set and/or their corresponding labels. Also called inversion.


###### 🛡️ Defensive actions 🛡️

  * Use of advanced cryptography. Countermeasures include [differential privacy](https://en.wikipedia.org/wiki/Differential_privacy), [homomorphic cryptography](https://en.wikipedia.org/wiki/Homomorphic_encryption) and [secure multiparty computation](https://en.wikipedia.org/wiki/Secure_multi-party_computation).

  * Use of regularization techniques such as [Dropout](https://en.wikipedia.org/wiki/Dilution_(neural_networks)) due to the relationship between overtraining and privacy.

  * [Model compression](https://medium.com/gsi-technology/an-overview-of-model-compression-techniques-for-deep-learning-in-space-3fd8d4ce84e5) has been proposed as a defense against reconstruction attacks.

###### 🔗 Useful links 🔗

  * [Membership Inference Attacks Against Machine Learning Models](https://arxiv.org/abs/1610.05820)
  * [Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures](https://dl.acm.org/doi/10.1145/2810103.2813677)
  * [Machine Learning Models that Remember Too Much](https://arxiv.org/abs/1709.07886)
  * [ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models](https://arxiv.org/abs/1806.01246)
  * [Deep Models Under the GAN: Information Leakage from Collaborative Deep Learning](https://arxiv.org/abs/1702.07464)
  * [LOGAN: Membership Inference Attacks Against Generative Models](https://petsymposium.org/popets/2019/popets-2019-0008.php)
  * [Overfitting, robustness, and malicious algorithms: A study of potential causes of privacy risk in machine learning](https://content.iospress.com/articles/journal-of-computer-security/jcs191362)
  * [Comprehensive Privacy Analysis of Deep Learning: Stand-alone and Federated Learning under Passive and Active White-box Inference Attacks](https://arxiv.org/abs/1812.00910)
  * [Inference Attacks Against Collaborative Learning](https://arxiv.org/abs/1805.04049)
  * [The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks](https://arxiv.org/abs/1802.08232)
  * [Towards the Science of Security and Privacy in Machine Learning](https://arxiv.org/abs/1611.03814)
  * [MemGuard: Defending against Black-Box Membership Inference Attacks via Adversarial Examples](https://arxiv.org/abs/1909.10594)
  * [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805)
  * [Property Inference Attacks on Fully Connected Neural Networks using Permutation Invariant Representations](https://dl.acm.org/doi/10.1145/3243734.3243834)
  * [Extracting Training Data from Diffusion Models](https://arxiv.org/abs/2301.13188)
  * [High-resolution image reconstruction with latent diffusion models from human brain activity](https://www.biorxiv.org/content/10.1101/2022.11.18.517004v1)
  * [Stealing and evading malware classifiers and antivirus at low false positive conditions](https://www.sciencedirect.com/science/article/abs/pii/S0167404823001025)
  * [Realistic fingerprint presentation attacks based on an adversarial approach](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10295513)
  * [Active Adversarial Tests](https://github.com/google-research/active-adversarial-tests): Increasing Confidence in Adversarial Robustness Evaluations. [![stars](https://badgen.net/github/stars/google-research/active-adversarial-tests)](https://github.com/google-research/active-adversarial-tests)
  * [GPT Jailbreak Status](https://github.com/tg12/gpt_jailbreak_status): Updates on the status of jailbreaking the OpenAI GPT language model. [![stars](https://badgen.net/github/stars/tg12/gpt_jailbreak_status)](https://github.com/tg12/gpt_jailbreak_status)
  * [Order of Magnitude Speedups for LLM Membership Inference](https://arxiv.org/abs/2409.14513)

##### 💉 Poisoning 💉

They aim to corrupt the training set by causing a machine-learning model to reduce its accuracy.

![Poisoning attack](/assets/images/poisoning.png)

This attack is difficult to detect when performed on the training data since the attack can propagate among different models using the same training data.

The adversary seeks to destroy the availability of the model by modifying the decision boundary and, as a result, producing incorrect predictions or, create a backdoor in a model. In the latter, the model behaves correctly (returning the desired predictions) in most cases, except for certain inputs specially created by the adversary that produce undesired results. The adversary can manipulate the results of the predictions and launch future attacks.

##### 🔓 Backdoors 🔓

[BadNets](https://arxiv.org/abs/1708.06733) are the simplest type of backdoor in a machine learning model. Moreover, BadNets are able to be preserved in a model, even if they are retrained again for a different task than the original model (transfer learning).

It is important to note that **public pre-trained models may contain backdoors**.

###### 🛡️ Defensive actions 🛡️

  * Detection of poisoned data, along with the use of data sanitization.

  * Robust training methods.

  * Specific defenses.
    * [Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks](https://ieeexplore.ieee.org/document/8835365)
    * [STRIP: A Defence Against Trojan Attacks on Deep Neural Networks](https://arxiv.org/abs/1902.06531)
    * [Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering](https://arxiv.org/abs/1811.03728)
    * [ABS: Scanning Neural Networks for Back-doors by Artificial Brain Stimulation](https://dl.acm.org/doi/10.1145/3319535.3363216)
    * [DeepInspect: A Black-box Trojan Detection and Mitigation Framework for Deep Neural Networks](https://www.ijcai.org/proceedings/2019/647)
    * [Defending Neural Backdoors via Generative Distribution Modeling](https://arxiv.org/abs/1910.04749)
    * [A Comprehensive Survey on Backdoor Attacks and Their Defenses in Face Recognition Systems](https://ieeexplore.ieee.org/document/10480615)
    * [DataElixir: Purifying Poisoned Dataset to Mitigate Backdoor Attacks via Diffusion Models](https://arxiv.org/abs/2312.11057)

###### 🔗 Useful links 🔗

  * [Poisoning Attacks against Support Vector Machines](https://arxiv.org/abs/1206.6389)
  * [Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning](https://arxiv.org/abs/1712.05526)
  * [Trojaning Attack on Neural Networks](https://www.semanticscholar.org/paper/Trojaning-Attack-on-Neural-Networks-Liu-Ma/08f7ac64b420210aa46fcbbdb0f206215f2e0644)
  * [Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks](https://arxiv.org/abs/1805.12185)
  * [Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks](https://arxiv.org/abs/1804.00792)
  * [Spectral Signatures in Backdoor Attacks](https://arxiv.org/abs/1811.00636)
  * [Latent Backdoor Attacks on Deep Neural Networks](https://dl.acm.org/doi/10.1145/3319535.3354209)
  * [Regula Sub-rosa: Latent Backdoor Attacks on Deep Neural Networks](https://arxiv.org/abs/1905.10447)
  * [Hidden Trigger Backdoor Attacks](https://arxiv.org/abs/1910.00033)
  * [Transferable Clean-Label Poisoning Attacks on Deep Neural Nets](https://arxiv.org/abs/1905.05897)
  * [TABOR: A Highly Accurate Approach to Inspecting and Restoring Trojan Backdoors in AI Systems](https://arxiv.org/abs/1908.01763)
  * [Towards Poisoning of Deep Learning Algorithms with Back-gradient Optimization](https://arxiv.org/abs/1708.08689)
  * [When Does Machine Learning FAIL? Generalized Transferability for Evasion and Poisoning Attacks](https://arxiv.org/abs/1803.06975)
  * [Certified Defenses for Data Poisoning Attacks](https://arxiv.org/abs/1706.03691)
  * [Input-Aware Dynamic Backdoor Attack](https://arxiv.org/abs/2010.08138)
  * [How To Backdoor Federated Learning](https://arxiv.org/abs/1807.00459)
  * [Planting Undetectable Backdoors in Machine Learning Models](https://arxiv.org/abs/2204.06974)
  * [Fool the AI!](https://fooltheai.mybluemix.net/): Hackers can use backdoors to poison training data and cause an AI model to misclassify images. Learn how IBM researchers can tell when data has been poisoned, and then guess what backdoors have been hidden in these datasets. Can you guess the backdoor?
  * [Backdoor Toolbox](https://github.com/vtu81/backdoor-toolbox): A compact toolbox for backdoor attacks and defenses. [![stars](https://badgen.net/github/stars/vtu81/backdoor-toolbox)](https://github.com/vtu81/backdoor-toolbox)

##### 🏃‍♂️ Evasion 🏃‍♂️

An adversary adds a small perturbation (in the form of noise) to the input of a machine learning model to make it classify incorrectly (example adversary).

![Evasion attack](/assets/images/evasion.png)

They are similar to poisoning attacks, but their main difference is that evasion attacks try to exploit the weaknesses of the model in the inference phase.

The goal of the adversary is for adversarial examples to be imperceptible to a human.

Two types of attack can be performed depending on the output desired by the opponent:

  * **Targeted**: the adversary aims to obtain a prediction of his choice.

    ![Targeted attack](/assets/images/targeted.png)

  * **Untargeted**: the adversary intends to achieve a misclassification.

    ![Untargeted attack](/assets/images/untargeted.png)

The most common attacks are **white-box attacks**:

  * [L-BFGS](https://arxiv.org/abs/1312.6199)
  * [FGSM](https://arxiv.org/abs/1412.6572)
  * [BIM](https://arxiv.org/abs/1607.02533)
  * [JSMA](https://arxiv.org/abs/1511.07528)
  * [Carlini & Wagner (C&W)](https://arxiv.org/abs/1608.04644)
  * [NewtonFool](https://andrewxiwu.github.io/public/papers/2017/JWJ17-objective-metrics-and-gradient-descent-based-algorithms-for-adversarial-examples-in-machine-learning.pdf)
  * [EAD](https://arxiv.org/abs/1709.04114)
  * [UAP](https://arxiv.org/abs/1610.08401)

###### 🛡️ Defensive actions 🛡️

  * Adversarial training, which consists of crafting adversarial examples during training to allow the model to learn features of the adversarial examples, making the model more robust to this type of attack.

  * Transformations on inputs.

  * Gradient masking/regularization. [Not very effective](https://arxiv.org/abs/1802.00420).

  * Weak defenses.

  * [Prompt Injection Defenses](https://github.com/tldrsec/prompt-injection-defenses): Every practical and proposed defense against prompt injection. [![stars](https://badgen.net/github/stars/tldrsec/prompt-injection-defenses)](https://github.com/tldrsec/prompt-injection-defenses)

  * [Lakera PINT Benchmark](https://github.com/lakeraai/pint-benchmark): The Prompt Injection Test (PINT) Benchmark provides a neutral way to evaluate the performance of a prompt injection detection system, like Lakera Guard, without relying on known public datasets that these tools can use to optimize for evaluation performance. [![stars](https://badgen.net/github/stars/lakeraai/pint-benchmark)](https://github.com/lakeraai/pint-benchmark)

  * [Devil's Inference](https://github.com/AI-Voodoo/Devil_Inference): A method to adversarially assess the Phi-3 Instruct model by observing the attention distribution across its heads when exposed to specific inputs. This approach prompts the model to adopt the 'devil's mindset’, enabling it to generate outputs of a violent nature. [![stars](https://badgen.net/github/stars/AI-Voodoo/Devil_Inference)](https://github.com/AI-Voodoo/Devil_Inference)

###### 🔗 Useful links 🔗

  * [Practical Black-Box Attacks against Machine Learning](https://arxiv.org/abs/1602.02697)
  * [The Limitations of Deep Learning in Adversarial Settings](https://arxiv.org/abs/1511.07528)
  * [Towards Evaluating the Robustness of Neural Networks](https://arxiv.org/abs/1608.04644)
  * [Distillation as a Defense to Adversarial Perturbations Against Deep Neural Networks](https://arxiv.org/abs/1511.04508)
  * [Adversarial examples in the physical world](https://arxiv.org/abs/1607.02533)
  * [Ensemble Adversarial Training: Attacks and Defenses](https://arxiv.org/abs/1705.07204)
  * [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)
  * [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199)
  * [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
  * [Delving into Transferable Adversarial Examples and Black-box Attacks](https://arxiv.org/abs/1611.02770)
  * [Adversarial machine learning at scale](https://arxiv.org/abs/1611.01236)
  * [Black-box Adversarial Attacks with Limited Queries and Information](https://arxiv.org/abs/1804.08598)
  * [Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks](https://arxiv.org/abs/1704.01155)
  * [Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models](https://arxiv.org/abs/1712.04248)
  * [Boosting Adversarial Attacks with Momentum](https://openaccess.thecvf.com/content_cvpr_2018/papers/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.pdf)
  * [The Space of Transferable Adversarial Examples](https://arxiv.org/abs/1704.03453)
  * [Countering Adversarial Images using Input Transformations](https://arxiv.org/abs/1711.00117)
  * [Defense-GAN: Protecting Classifiers Against Adversarial Attacks Using Generative Models](https://arxiv.org/abs/1805.06605)
  * [Synthesizing Robust Adversarial Examples](https://arxiv.org/abs/1707.07397)
  * [Mitigating adversarial effects through randomization](https://arxiv.org/abs/1711.01991)
  * [On Detecting Adversarial Perturbations](https://arxiv.org/abs/1702.04267)
  * [Adversarial Patch](https://arxiv.org/abs/1712.09665)
  * [PixelDefend: Leveraging Generative Models to Understand and Defend against Adversarial Examples](https://arxiv.org/abs/1710.10766)
  * [One Pixel Attack for Fooling Deep Neural Networks](https://arxiv.org/abs/1710.08864)
  * [Efficient Defenses Against Adversarial Attacks](https://arxiv.org/abs/1707.06728)
  * [Robust Physical-World Attacks on Deep Learning Visual Classification](https://ieeexplore.ieee.org/document/8578273)
  * [Adversarial Perturbations Against Deep Neural Networks for Malware Classification](https://arxiv.org/abs/1606.04435)
  * [3D Adversarial Attacks Beyond Point Cloud](https://arxiv.org/abs/2104.12146)
  * [Adversarial Perturbations Fool Deepfake Detectors](https://arxiv.org/abs/2003.10596)
  * [Adversarial Deepfakes: Evaluating Vulnerability of Deepfake Detectors to Adversarial Examples](https://arxiv.org/abs/2002.12749)
  * [An Overview of Vulnerabilities of Voice Controlled Systems](https://arxiv.org/abs/1803.09156)
  * [FastWordBug: A Fast Method To Generate Adversarial Text Against NLP Applications](https://arxiv.org/abs/2002.00760)
  * [Phantom of the ADAS: Securing Advanced Driver Assistance Systems from Split-Second Phantom Attacks](https://www.nassiben.com/phantoms)
  * [llm-attacks](https://github.com/llm-attacks/llm-attacks): Universal and Transferable Attacks on Aligned Language Models. [![stars](https://badgen.net/github/stars/llm-attacks/llm-attacks)](https://github.com/llm-attacks/llm-attacks)
  * [Attacks on AI Models: Prompt Injection vs supply chain poisoning](https://blog.mithrilsecurity.io/attacks-on-ai-models-prompt-injection-vs-supply-chain-poisoning/)
  * [Prompt Injection attack against LLM-integrated Applications](https://arxiv.org/pdf/2306.05499.pdf)
  * [garak](https://github.com/leondz/garak): LLM vulnerability scanner. [![stars](https://badgen.net/github/stars/leondz/garak)](https://github.com/leondz/garak)
  * [Simple Adversarial Transformations in PyTorch](https://davidstutz.de/simple-adversarial-transformations-in-pytorch/)
  * [ChatGPT Plugins: Data Exfiltration via Images & Cross Plugin Request Forgery](https://embracethered.com/blog/posts/2023/chatgpt-webpilot-data-exfil-via-markdown-injection/)
  * [Image Hijacks: Adversarial Images can Control Generative Models at Runtime](https://arxiv.org/abs/2309.00236)
  * [Multi-attacks: Many images + the same adversarial attack → many target labels](https://arxiv.org/abs/2308.03792)
  * [ACTIVE: Towards Highly Transferable 3D Physical Camouflage for Universal and Robust Vehicle Evasion](https://arxiv.org/abs/2308.07009)
  * [LLM Red Teaming GPTS's: Prompt Leaking, API Leaking, Documents Leaking](https://adversa.ai/blog/llm-red-teaming-gpts-prompt-leaking-api-leaking-documents-leaking/)
  * [Human-Producible Adversarial Examples](https://arxiv.org/abs/2310.06474)
  * [Multilingual Jailbreak Challenges in Large Language Models](https://arxiv.org/abs/2310.06474)
  * [Misusing Tools in Large Language Models With Visual Adversarial Examples](https://arxiv.org/abs/2310.03185)
  * [AutoDAN: Interpretable Gradient-Based Adversarial Attacks on Large Language Models](https://arxiv.org/abs/2310.15140)
  * [Multimodal Injection](https://github.com/ebagdasa/multimodal_injection): (Ab)using Images and Sounds for Indirect Instruction Injection in Multi-Modal LLMs. [![stars](https://badgen.net/github/stars/ebagdasa/multimodal_injection)](https://github.com/ebagdasa/multimodal_injection)
  * [JailbreakingLLMs](https://github.com/patrickrchao/JailbreakingLLMs): Jailbreaking Black Box Large Language Models in Twenty Queries. [![stars](https://badgen.net/github/stars/patrickrchao/JailbreakingLLMs)](https://github.com/patrickrchao/JailbreakingLLMs)
  * [Tree of Attacks: Jailbreaking Black-Box LLMs Automatically](https://assets-global.website-files.com/62a8db3f7f80ab5d3420c03a/656eaaed8e762c7543693902_Robust_Intelligence_Blackbox_Attacks_on_LLMs.pdf)
  * [GPTs](https://github.com/linexjlin/GPTs): Leaked prompts of GPTs. [![stars](https://badgen.net/github/stars/linexjlin/GPTs)](https://github.com/linexjlin/GPTs)
  * [AI Exploits](https://github.com/protectai/ai-exploits): A collection of real world AI/ML exploits for responsibly disclosed vulnerabilities. [![stars](https://badgen.net/github/stars/protectai/ai-exploits)](https://github.com/protectai/ai-exploits)
  * [LLM Agents can Autonomously Hack Websites](https://arxiv.org/abs/2402.06664v1)
  * [Cloudflare announces Firewall for AI](https://blog.cloudflare.com/firewall-for-ai)
  * [PromptInject](https://github.com/agencyenterprise/PromptInject): Framework that assembles prompts in a modular fashion to provide a quantitative analysis of the robustness of LLMs to adversarial prompt attacks. [![stars](https://badgen.net/github/stars/agencyenterprise/PromptInject)](https://github.com/agencyenterprise/PromptInject)
  * [LLM Red Teaming: Adversarial, Programming, and Linguistic approaches VS ChatGPT, Claude, Mistral, Grok, LLAMA, and Gemini](https://adversa.ai/blog/llm-red-teaming-vs-grok-chatgpt-claude-gemini-bing-mistral-llama/)
  * [The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions](https://arxiv.org/abs/2404.13208)
  * [Prompt Injection / JailBreaking a Banking LLM Agent (GPT-4, Langchain)](https://www.youtube.com/watch?v=5rXVg8cxne4)
  * [GitHub Copilot Chat: From Prompt Injection to Data Exfiltration](https://embracethered.com/blog/posts/2024/github-copilot-chat-prompt-injection-data-exfiltration/?s=35)
  * [Adversarial Examples are Misaligned in Diffusion Model Manifolds](https://arxiv.org/abs/2401.06637)
  * [Image-to-Text Logic Jailbreak: Your Imagination Can Help You Do Anything](https://arxiv.org/abs/2407.02534)
  * [Mitigating Skeleton Key, a new type of generative AI jailbreak technique](https://www.microsoft.com/en-us/security/blog/2024/06/26/mitigating-skeleton-key-a-new-type-of-generative-ai-jailbreak-technique/)
  * [Image Obfuscation Benchmark](https://github.com/google-deepmind/image_obfuscation_benchmark): This repository contains the code to evaluate models on the image obfuscation benchmark, first presented in [Benchmarking Robustness to Adversarial Image Obfuscations](https://arxiv.org/abs/2301.12993). [![stars](https://badgen.net/github/stars/google-deepmind/image_obfuscation_benchmark)](https://github.com/google-deepmind/image_obfuscation_benchmark)
  * [Jailbreaking Large Language Models with Symbolic Mathematics](https://arxiv.org/abs/2409.11445)


#### 🛠️ Tools 🛠️

| Name | Type | Supported algorithms | Supported attack types | Attack/Defence | Supported frameworks | Popularity |
| ---------- | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
| [Cleverhans](https://github.com/cleverhans-lab/cleverhans) | Image | [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning) | Evasion | Attack | [Tensorflow](https://www.tensorflow.org), [Keras](https://keras.io), [JAX](https://github.com/google/jax) | [![stars](https://badgen.net/github/stars/cleverhans-lab/cleverhans)](https://github.com/cleverhans-lab/cleverhans)|
| [Foolbox](https://github.com/bethgelab/foolbox) | Image | Deep Learning | Evasion | Attack | Tensorflow, [PyTorch](https://pytorch.org), JAX | [![stars](https://badgen.net/github/stars/bethgelab/foolbox)](https://github.com/bethgelab/foolbox)|
| [ART](https://github.com/Trusted-AI/adversarial-robustness-toolbox) | Any type (image, tabular data, audio,...) | Deep Learning, [SVM](https://en.wikipedia.org/wiki/Support_vector_machine), [LR](https://en.wikipedia.org/wiki/Logistic_regression), etc. | Any (extraction, inference, poisoning, evasion) | Both | Tensorflow, Keras, Pytorch, [Scikit Learn](https://scikit-learn.org) | [![stars](https://badgen.net/github/stars/Trusted-AI/adversarial-robustness-toolbox)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)|
| [TextAttack](https://github.com/QData/TextAttack) | Text | Deep Learning | Evasion | Attack | Keras, [HuggingFace](https://huggingface.co/) | [![stars](https://badgen.net/github/stars/QData/TextAttack)](https://github.com/QData/TextAttack)|
| [Advertorch](https://github.com/BorealisAI/advertorch) | Image | Deep Learning | Evasion | Both | --- | [![stars](https://badgen.net/github/stars/BorealisAI/advertorch)](https://github.com/BorealisAI/advertorch)|
| [AdvBox](https://github.com/advboxes/AdvBox) | Image | Deep Learning | Evasion | Both | PyTorch, Tensorflow, [MxNet](https://mxnet.apache.org) | [![stars](https://badgen.net/github/stars/advboxes/AdvBox)](https://github.com/advboxes/AdvBox)|
| [DeepRobust](https://github.com/DSE-MSU/DeepRobust) | Image, graph | Deep Learning | Evasion | Both | PyTorch | [![stars](https://badgen.net/github/stars/DSE-MSU/DeepRobust)](https://github.com/DSE-MSU/DeepRobust)|
| [Counterfit](https://github.com/Azure/counterfit) | Any | Any | Evasion | Attack | --- | [![stars](https://badgen.net/github/stars/Azure/counterfit)](https://github.com/Azure/counterfit)|
| [Adversarial Audio Examples](https://github.com/carlini/audio_adversarial_examples) | Audio | [DeepSpeech](https://github.com/mozilla/DeepSpeech) | Evasion | Attack | --- | [![stars](https://badgen.net/github/stars/carlini/audio_adversarial_examples)](https://github.com/carlini/audio_adversarial_examples)|

###### ART

[Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox), abbreviated as ART, is an open-source Adversarial Machine Learning library for testing the robustness of machine learning models.

![ART logo](/assets/images/art.png)

It is developed in Python and implements extraction, inversion, poisoning and evasion attacks and defenses.

ART supports the most popular frameworks: Tensorflow, Keras, PyTorch, MxNet, and ScikitLearn among many others.

It is not limited to the use of models that use images as input but also supports other types of data, such as audio, video, tabular data, etc.

> [Workshop to learn Adversarial Machine Learning with ART 🇪🇸](https://github.com/jiep/adversarial-machine-learning)

###### Cleverhans

[Cleverhans](https://github.com/cleverhans-lab/cleverhans) is a library for performing evasion attacks and testing the robustness of a deep learning model on image models.

![Cleverhans logo](/assets/images/cleverhans.png)

It is developed in Python and integrates with the Tensorflow, Torch and JAX frameworks.

It implements numerous attacks such as L-BFGS, FGSM, JSMA, C&W, among others.

## 🔧 Use 🔧

AI is used to accomplish malicious tasks and boost classic attacks.

### 🕵️‍♂️ Pentesting 🕵️‍♂️

  * [GyoiThon](https://github.com/gyoisamurai/GyoiThon): Next generation penetration test tool, intelligence gathering tool for web server. [![stars](https://badgen.net/github/stars/gyoisamurai/GyoiThon)](https://github.com/gyoisamurai/GyoiThon)
  * [Deep Exploit](https://github.com/13o-bbr-bbq/machine_learning_security/tree/master/DeepExploit): Fully automatic penetration test tool using Deep Reinforcement Learning. [![stars](https://badgen.net/github/stars/13o-bbr-bbq/machine_learning_security)](https://github.com/13o-bbr-bbq/machine_learning_security)
  * [AutoPentest-DRL](https://github.com/crond-jaist/AutoPentest-DRL): Automated penetration testing using deep reinforcement learning. [![stars](https://badgen.net/github/stars/crond-jaist/AutoPentest-DRL)](https://github.com/crond-jaist/AutoPentest-DRL)
  * [DeepGenerator](https://github.com/13o-bbr-bbq/machine_learning_security/tree/master/Generator): Fully automatically generate injection codes for web application assessment using Genetic Algorithm and Generative Adversarial Networks.
  * [Eyeballer](https://github.com/BishopFox/eyeballer): Eyeballer is meant for large-scope network penetration tests where you need to find "interesting" targets from a huge set of web-based hosts. [![stars](https://badgen.net/github/stars/BishopFox/eyeballer)](https://github.com/BishopFox/eyeballer)
  * [Nebula](https://github.com/berylliumsec/nebula): AI-Powered Ethical Hacking Assistant. [![stars](https://badgen.net/github/stars/berylliumsec/nebula)](https://github.com/berylliumsec/nebula)
  * [Teams of LLM Agents can Exploit Zero-Day Vulnerabilities](https://arxiv.org/abs/2406.01637)
  * [Insights and Current Gaps in Open-Source LLM Vulnerability Scanners: A Comparative Analysis](https://arxiv.org/abs/2410.16527)

### 🦠 Malware 🦠

  * [DeepLocker](https://i.blackhat.com/us-18/Thu-August-9/us-18-Kirat-DeepLocker-Concealing-Targeted-Attacks-with-AI-Locksmithing.pdf): Concealing targeted attacks with AI locksmithing, by IBM Labs on BH.
  * [An Overview of Artificial Intelligence Used in Malware](https://link.springer.com/chapter/10.1007/978-3-031-17030-0_4): A curated list of AI Malware resources.
  * [DeepObfusCode](https://arxiv.org/abs/1909.01837): Source code obfuscation through sequence-to-sequence networks.
  * [AutoCAT](https://arxiv.org/abs/2208.08025): Reinforcement learning for automated exploration of cache-timing attacks.
  * [AI-BASED BOTNET](https://arxiv.org/abs/2112.02223): A game-theoretic approach for AI-based botnet attack defence.
  * [SECML_Malware](https://github.com/pralab/secml_malware): Python library for creating adversarial attacks against Windows Malware detectors. [![stars](https://badgen.net/github/stars/pralab/secml_malware)](https://github.com/pralab/secml_malware)
  * [Transcendent-release](https://github.com/s2labres/transcendent-release): Using conformal evaluation to detect concept drift affecting malware detection. [![stars](https://badgen.net/github/stars/s2labres/transcendent-release)](https://github.com/s2labres/transcendent-release)

### 🗺️ OSINT 🗺️

  * [SNAP_R](https://github.com/zerofox-oss/SNAP_R): Generate automatically spear-phishing posts on social media. [![stars](https://badgen.net/github/stars/zerofox-oss/SNAP_R)](https://github.com/zerofox-oss/SNAP_R)
  * [SpyScrap](https://github.com/RuthGnz/SpyScrap): SpyScrap combines facial recognition methods to filter the results and uses natural language processing to obtain important entities from the website the user appears. [![stars](https://badgen.net/github/stars/RuthGnz/SpyScrap)](https://github.com/RuthGnz/SpyScrap)

### 📧 Phishing 📧

  * [DeepDGA](https://github.com/roreagan/DeepDGA): Implementation of DeepDGA: Adversarially-Tuned Domain Generation and Detection. [![stars](https://badgen.net/github/stars/roreagan/DeepDGA)](https://github.com/roreagan/DeepDGA)

### 🕵 Threat Intelligence 🕵

  * [From Sands to Mansions: Enabling Automatic Full-Life-Cycle Cyberattack Construction with LLM](https://arxiv.org/abs/2407.16928) 


### 🌀 Side channels 🌀

  * [SCAAML](https://github.com/google/scaaml): Side Channel Attacks Assisted with Machine Learning. [![stars](https://badgen.net/github/stars/google/scaaml)](https://github.com/google/scaaml)

### 👨‍🎤 Generative AI 👨‍🎤

#### 🔊 Audio 🔊

##### 🛠️ Tools 🛠️
  * [deep-voice-conversion](https://github.com/andabi/deep-voice-conversion): Deep neural networks for voice conversion (voice style transfer) in Tensorflow. [![stars](https://badgen.net/github/stars/andabi/deep-voice-conversion)](https://github.com/andabi/deep-voice-conversion)
  * [tacotron](https://github.com/keithito/tacotron): A TensorFlow implementation of Google's Tacotron speech synthesis with pre-trained model (unofficial). [![stars](https://badgen.net/github/stars/keithito/tacotron)](https://github.com/keithito/tacotron)
  * [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning): Clone a voice in 5 seconds to generate arbitrary speech in real-time. [![stars](https://badgen.net/github/stars/CorentinJ/Real-Time-Voice-Cloning)](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
  * [mimic2](https://github.com/MycroftAI/mimic2): Text to Speech engine based on the Tacotron architecture, initially implemented by Keith Ito. [![stars](https://badgen.net/github/stars/MycroftAI/mimic2)](https://github.com/MycroftAI/mimic2)
  * [Neural-Voice-Cloning-with-Few-Samples](https://github.com/Sharad24/Neural-Voice-Cloning-with-Few-Samples): Implementation of Neural Voice Cloning with Few Samples Research Paper by Baidu. [![stars](https://badgen.net/github/stars/Sharad24/Neural-Voice-Cloning-with-Few-Samples)](https://github.com/Sharad24/Neural-Voice-Cloning-with-Few-Samples)
  * [Vall-E](https://github.com/enhuiz/vall-e): An unofficial PyTorch implementation of the audio LM VALL-E. [![stars](https://badgen.net/github/stars/enhuiz/vall-e)](https://github.com/enhuiz/vall-e)
  * [voice-changer](https://github.com/w-okada/voice-changer): Realtime Voice Changer. [![stars](https://badgen.net/github/stars/w-okada/voice-changer)](https://github.com/w-okada/voice-changer)
  * [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI): An easy-to-use Voice Conversion framework based on VITS. [![stars](https://badgen.net/github/stars/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
  * [Audiocraft](https://github.com/facebookresearch/audiocraft): Audiocraft is a library for audio processing and generation with deep learning. It features the state-of-the-art EnCodec audio compressor/tokenizer, along with MusicGen, a simple and controllable music generation LM with textual and melodic conditioning. [![stars](https://badgen.net/github/stars/facebookresearch/audiocraft)](https://github.com/facebookresearch/audiocraft)
  * [VALL-E-X](https://github.com/Plachtaa/VALL-E-X): An open source implementation of Microsoft's VALL-E X zero-shot TTS model. [![stars](https://badgen.net/github/stars/Plachtaa/VALL-E-X)](https://github.com/Plachtaa/VALL-E-X)
  * [OpenVoice](https://github.com/myshell-ai/OpenVoice): Instant voice cloning by MyShell. [![stars](https://badgen.net/github/stars/myshell-ai/OpenVoice)](https://github.com/myshell-ai/OpenVoice)
  * [MeloTTS](https://github.com/myshell-ai/MeloTTS): High-quality multi-lingual text-to-speech library by MyShell.ai. Support English, Spanish, French, Chinese, Japanese and Korean. [![stars](https://badgen.net/github/stars/myshell-ai/MeloTTS)](https://github.com/myshell-ai/MeloTTS)
  * [VoiceCraft](https://github.com/jasonppy/VoiceCraft): Zero-Shot Speech Editing and Text-to-Speech in the Wild. [![stars](https://badgen.net/github/stars/jasonppy/VoiceCraft)](https://github.com/jasonppy/VoiceCraft)
  * [Parler-TTS](https://github.com/huggingface/parler-tts): Inference and training library for high-quality TTS models. [![stars](https://badgen.net/github/stars/huggingface/parler-tts)](https://github.com/huggingface/parler-tts)
  * [ChatTTS](https://github.com/2noise/ChatTTS): A generative speech model for daily dialogue. [![stars](https://badgen.net/github/stars/2noise/ChatTTS)](https://github.com/2noise/ChatTTS)



##### 💡 Applications 💡

  * [Lip2Wav](https://github.com/Rudrabha/Lip2Wav): Generate high quality speech from only lip movements. [![stars](https://badgen.net/github/stars/Rudrabha/Lip2Wav)](https://github.com/Rudrabha/Lip2Wav)
  * [AudioLDM: Text-to-Audio Generation with Latent Diffusion Models](https://huggingface.co/spaces/haoheliu/audioldm-text-to-audio-generation)
  * [deepvoice3_pytorch](https://github.com/r9y9/deepvoice3_pytorch): PyTorch implementation of convolutional neural networks-based text-to-speech synthesis models. [![stars](https://badgen.net/github/stars/r9y9/deepvoice3_pytorch)](https://github.com/r9y9/deepvoice3_pytorch)
  * [🎸 Riffusion](https://github.com/riffusion/riffusion): Stable diffusion for real-time music generation. [![stars](https://badgen.net/github/stars/riffusion/riffusion)](https://github.com/riffusion/riffusion)
  * [whisper.cpp](https://github.com/ggerganov/whisper.cpp): Port of OpenAI's Whisper model in C/C++. [![stars](https://badgen.net/github/stars/ggerganov/whisper.cpp)](https://github.com/ggerganov/whisper.cpp)
  * [TTS](https://github.com/coqui-ai/TTS): 🐸💬 - a deep learning toolkit for Text-to-Speech, battle-tested in research and production. [![stars](https://badgen.net/github/stars/coqui-ai/TTS)](https://github.com/coqui-ai/TTS)
  * [YourTTS](https://github.com/Edresson/YourTTS): Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for everyone. [![stars](https://badgen.net/github/stars/Edresson/YourTTS)](https://github.com/Edresson/YourTTS)
  * [TorToiSe](https://github.com/neonbjb/tortoise-tts): A multi-voice TTS system trained with an emphasis on quality. [![stars](https://badgen.net/github/stars/neonbjb/tortoise-tts)](https://github.com/neonbjb/tortoise-tts)
  * [DiffSinger](https://github.com/MoonInTheRiver/DiffSinger): Singing Voice Synthesis via Shallow Diffusion Mechanism (SVS & TTS). [![stars](https://badgen.net/github/stars/MoonInTheRiver/DiffSinger)](https://github.com/MoonInTheRiver/DiffSinger)
  * [WaveNet vocoder](https://github.com/r9y9/wavenet_vocoder): Implementation of the WaveNet vocoder, which can generate high-quality raw speech samples conditioned on linguistic or acoustic features. [![stars](https://badgen.net/github/stars/r9y9/wavenet_vocoder)](https://github.com/r9y9/wavenet_vocoder)
  * [Deepvoice3_pytorch](https://github.com/r9y9/deepvoice3_pytorch): PyTorch implementation of convolutional neural networks-based text-to-speech synthesis models. [![stars](https://badgen.net/github/stars/r9y9/deepvoice3_pytorch)](https://github.com/r9y9/deepvoice3_pytorch)
  * [eSpeak NG Text-to-Speech](https://github.com/espeak-ng/espeak-ng): eSpeak NG is an open source speech synthesizer that supports more than hundred languages and accents. [![stars](https://badgen.net/github/stars/espeak-ng/espeak-ng)](https://github.com/espeak-ng/espeak-ng)
  * [RealChar](https://github.com/Shaunwei/RealChar): Create, Customize and Talk to your AI Character/Companion in Realtime (All in One Codebase!). Have a natural seamless conversation with AI everywhere (mobile, web and terminal) using LLM OpenAI GPT3.5/4, Anthropic Claude2, Chroma Vector DB, Whisper Speech2Text, ElevenLabs Text2Speech. [![stars](https://badgen.net/github/stars/Shaunwei/RealChar)](https://github.com/Shaunwei/RealChar)
  * [Neural Voice Cloning with a Few Samples](https://proceedings.neurips.cc/paper/2018/hash/4559912e7a94a9c32b09d894f2bc3c82-Abstract.html)
  * [NAUTILUS: A Versatile Voice Cloning System](https://ieeexplore.ieee.org/abstract/document/9246264)
  * [Learning to Speak Fluently in a Foreign Language: Multilingual Speech Synthesis and Cross-Language Voice Cloning](https://arxiv.org/abs/1907.04448)
  * [When Good Becomes Evil: Keystroke Inference with Smartwatch](https://dl.acm.org/doi/10.1145/2810103.2813668)
  * [KeyListener: Inferring Keystrokes on QWERTY Keyboard of Touch Screen through Acoustic Signals](https://ieeexplore.ieee.org/document/8737591)
  * [This Voice Does Not Exist: On Voice Synthesis, Audio Deepfakes and Their Detection](https://this-voice-does-not-exist.com)
  * [AudioSep](https://github.com/Audio-AGI/AudioSep): Separate Anything You Describe. [![stars](https://badgen.net/github/stars/Audio-AGI/AudioSep)](https://github.com/Audio-AGI/AudioSep)
  * [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools): Generative models for conditional audio generation. [![stars](https://badgen.net/github/stars/Stability-AI/stable-audio-tools)](https://github.com/Stability-AI/stable-audio-tools)
  * [GPT-SoVITS-WebUI](https://github.com/RVC-Boss/GPT-SoVITS): 1 min voice data can also be used to train a good TTS model! (few shot voice cloning). [![stars](https://badgen.net/github/stars/RVC-Boss/GPT-SoVITS)](https://github.com/RVC-Boss/GPT-SoVITS)
  * [Hybrid-Net](https://github.com/DoMusic/Hybrid-Net): Real-time audio source separation, generate lyrics, chords, beat. [![stars](https://badgen.net/github/stars/DoMusic/Hybrid-Net)](https://github.com/DoMusic/Hybrid-Net)
  * [CosyVoice](https://github.com/FunAudioLLM/CosyVoice): Multi-lingual large voice generation model, providing inference, training and deployment full-stack ability. [![stars](https://badgen.net/github/stars/FunAudioLLM/CosyVoice)](https://github.com/FunAudioLLM/CosyVoice)
  * [EasyVolcap](https://github.com/zju3dv/EasyVolcap): Accelerating Neural Volumetric Video Research. [![stars](https://badgen.net/github/stars/zju3dv/EasyVolcap)](https://github.com/zju3dv/EasyVolcap)



##### 🔎 Detection 🔎
  * [fake-voice-detection](https://github.com/dessa-oss/fake-voice-detection): Using temporal convolution to detect Audio Deepfakes. [![stars](https://badgen.net/github/stars/dessa-oss/fake-voice-detection)](https://github.com/dessa-oss/fake-voice-detection)
  * [A robust voice spoofing detection system using novel CLS-LBP features and LSTM](https://www.sciencedirect.com/science/article/pii/S1319157822000684)
  * [Voice spoofing detector: A unified anti-spoofing framework](https://www.sciencedirect.com/science/article/abs/pii/S0957417422002330)
  * [Securing Voice-Driven Interfaces Against Fake (Cloned) Audio Attacks](https://ieeexplore.ieee.org/abstract/document/8695320)
  * [DeepSonar: Towards Effective and Robust Detection of AI-Synthesized Fake Voices](https://dl.acm.org/doi/abs/10.1145/3394171.3413716)
  * [Fighting AI with AI: Fake Speech Detection Using Deep Learning](https://www.aes.org/e-lib/online/browse.cfm?elib=20479)
  * [A Review of Modern Audio Deepfake Detection Methods: Challenges and Future Directions](https://www.mdpi.com/1999-4893/15/5/155)

#### 📷 Image 📷

##### 🛠️ Tools 🛠️

  * [StyleGAN](https://github.com/NVlabs/stylegan): StyleGAN - Official TensorFlow Implementation. [![stars](https://badgen.net/github/stars/NVlabs/stylegan)](https://github.com/NVlabs/stylegan)
  * [StyleGAN2](https://github.com/NVlabs/stylegan2): StyleGAN2 - Official TensorFlow Implementation. [![stars](https://badgen.net/github/stars/NVlabs/stylegan2)](https://github.com/NVlabs/stylegan2)
  * [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch): StyleGAN2-ADA - Official PyTorch implementation. [![stars](https://badgen.net/github/stars/NVlabs/stylegan2-ada-pytorch)](https://github.com/NVlabs/stylegan2-ada-pytorch)
  * [StyleGAN-nada](https://github.com/rinongal/StyleGAN-nada): CLIP-Guided Domain Adaptation of Image Generators. [![stars](https://badgen.net/github/stars/rinongal/StyleGAN-nada)](https://github.com/rinongal/StyleGAN-nada)
  * [StyleGAN3](https://github.com/NVlabs/stylegan3): Official PyTorch implementation of StyleGAN3. [![stars](https://badgen.net/github/stars/NVlabs/stylegan3)](https://github.com/NVlabs/stylegan3)
  * [Imaginaire](https://github.com/NVlabs/imaginaire): Imaginaire is a pytorch library that contains the optimized implementation of several image and video synthesis methods developed at NVIDIA. [![stars](https://badgen.net/github/stars/NVlabs/imaginaire)](https://github.com/NVlabs/imaginaire)
  * [ffhq-dataset](https://github.com/NVlabs/ffhq-dataset): Flickr-Faces-HQ Dataset (FFHQ). [![stars](https://badgen.net/github/stars/NVlabs/ffhq-dataset)](https://github.com/NVlabs/ffhq-dataset)
  * [DALLE2-pytorch](https://github.com/lucidrains/DALLE2-pytorch): Implementation of DALL-E 2, OpenAI's updated text-to-image synthesis neural network, in Pytorch. [![stars](https://badgen.net/github/stars/lucidrains/DALLE2-pytorch)](https://github.com/lucidrains/DALLE2-pytorch)
  * [ImaginAIry](https://github.com/brycedrennan/imaginAIry): AI imagined images. Pythonic generation of stable diffusion images. [![stars](https://badgen.net/github/stars/brycedrennan/imaginAIry)](https://github.com/brycedrennan/imaginAIry)
  * [Lama Cleaner](https://github.com/Sanster/lama-cleaner): Image inpainting tool powered by SOTA AI Model. Remove any unwanted object, defect, or people from your pictures or erase and replace(powered by stable diffusion) anything on your pictures. [![stars](https://badgen.net/github/stars/Sanster/lama-cleaner)](https://github.com/Sanster/lama-cleaner)
  * [Invertible-Image-Rescaling](https://github.com/pkuxmq/Invertible-Image-Rescaling): This is the PyTorch implementation of paper: Invertible Image Rescaling. [![stars](https://badgen.net/github/stars/pkuxmq/Invertible-Image-Rescaling)](https://github.com/pkuxmq/Invertible-Image-Rescaling)
  * [DifFace](https://github.com/zsyOAOA/DifFace): Blind Face Restoration with Diffused Error Contraction (PyTorch). [![stars](https://badgen.net/github/stars/zsyOAOA/DifFace)](https://github.com/zsyOAOA/DifFace)
  * [CodeFormer](https://github.com/sczhou/CodeFormer): Towards Robust Blind Face Restoration with Codebook Lookup Transformer. [![stars](https://badgen.net/github/stars/sczhou/CodeFormer)](https://github.com/sczhou/CodeFormer)
  * [Custom Diffusion](https://github.com/adobe-research/custom-diffusion): Multi-Concept Customization of Text-to-Image Diffusion. [![stars](https://badgen.net/github/stars/adobe-research/custom-diffusion)](https://github.com/adobe-research/custom-diffusion)
  * [Diffusers](https://github.com/huggingface/diffusers): 🤗 Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch. [![stars](https://badgen.net/github/stars/huggingface/diffusers)](https://github.com/huggingface/diffusers)
  * [Stable Diffusion](https://github.com/Stability-AI/stablediffusion): High-Resolution Image Synthesis with Latent Diffusion Models. [![stars](https://badgen.net/github/stars/Stability-AI/stablediffusion)](https://github.com/Stability-AI/stablediffusion)
  * [InvokeAI](https://github.com/invoke-ai/InvokeAI): InvokeAI is a leading creative engine for Stable Diffusion models, empowering professionals, artists, and enthusiasts to generate and create visual media using the latest AI-driven technologies. The solution offers an industry-leading WebUI, supports terminal use through a CLI, and serves as the foundation for multiple commercial products. [![stars](https://badgen.net/github/stars/invoke-ai/InvokeAI)](https://github.com/invoke-ai/InvokeAI)
  * [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui): Stable Diffusion web UI. [![stars](https://badgen.net/github/stars/AUTOMATIC1111/stable-diffusion-webui)](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
  * [Stable Diffusion Infinity](https://github.com/lkwq007/stablediffusion-infinity): Outpainting with Stable Diffusion on an infinite canvas. [![stars](https://badgen.net/github/stars/lkwq007/stablediffusion-infinity)](https://github.com/lkwq007/stablediffusion-infinity)
  * [Fast Stable Diffusion](https://github.com/TheLastBen/fast-stable-diffusion): fast-stable-diffusion + DreamBooth. [![stars](https://badgen.net/github/stars/TheLastBen/fast-stable-diffusion)](https://github.com/TheLastBen/fast-stable-diffusion)
  * [GET3D](https://github.com/nv-tlabs/GET3D): A Generative Model of High Quality 3D Textured Shapes Learned from Images. [![stars](https://badgen.net/github/stars/nv-tlabs/GET3D)](https://github.com/nv-tlabs/GET3D)
  * [Awesome AI Art Image Synthesis](https://github.com/altryne/awesome-ai-art-image-synthesis): A list of awesome tools, ideas, prompt engineering tools, collabs, models, and helpers for the prompt designer playing with aiArt and image synthesis. Covers Dalle2, MidJourney, StableDiffusion, and open source tools. [![stars](https://badgen.net/github/stars/altryne/awesome-ai-art-image-synthesis)](https://github.com/altryne/awesome-ai-art-image-synthesis)
  * [Stable Diffusion](https://github.com/CompVis/stable-diffusion): A latent text-to-image diffusion model. [![stars](https://badgen.net/github/stars/CompVis/stable-diffusion)](https://github.com/CompVis/stable-diffusion)
  * [Weather Diffusion](https://github.com/IGITUGraz/WeatherDiffusion): Code for "Restoring Vision in Adverse Weather Conditions with Patch-Based Denoising Diffusion Models". [![stars](https://badgen.net/github/stars/IGITUGraz/WeatherDiffusion)](https://github.com/IGITUGraz/WeatherDiffusion)
  * [DF-GAN](https://github.com/tobran/DF-GAN): A Simple and Effective Baseline for Text-to-Image Synthesis. [![stars](https://badgen.net/github/stars/tobran/DF-GAN)](https://github.com/tobran/DF-GAN)
  * [Dall-E Playground](https://github.com/saharmor/dalle-playground): A playground to generate images from any text prompt using Stable Diffusion (past: using DALL-E Mini). [![stars](https://badgen.net/github/stars/saharmor/dalle-playground)](https://github.com/saharmor/dalle-playground)
  * [MM-CelebA-HQ-Dataset](https://github.com/IIGROUP/MM-CelebA-HQ-Dataset): A large-scale face image dataset that allows text-to-image generation, text-guided image manipulation, sketch-to-image generation, GANs for face generation and editing, image caption, and VQA. [![stars](https://badgen.net/github/stars/IIGROUP/MM-CelebA-HQ-Dataset)](https://github.com/IIGROUP/MM-CelebA-HQ-Dataset)
  * [Deep Daze](https://github.com/lucidrains/deep-daze): Simple command line tool for text-to-image generation using OpenAI's CLIP and Siren (Implicit neural representation network). [![stars](https://badgen.net/github/stars/lucidrains/deep-daze)](https://github.com/lucidrains/deep-daze)
  * [StyleMapGAN](https://github.com/naver-ai/StyleMapGAN): Exploiting Spatial Dimensions of Latent in GAN for Real-time Image Editing. [![stars](https://badgen.net/github/stars/naver-ai/StyleMapGAN)](https://github.com/naver-ai/StyleMapGAN)
  * [Kandinsky-2](https://github.com/ai-forever/Kandinsky-2): Multilingual text2image latent diffusion model. [![stars](https://badgen.net/github/stars/ai-forever/Kandinsky-2)](https://github.com/ai-forever/Kandinsky-2)
  * [DragGAN](https://github.com/XingangPan/DragGAN): Interactive Point-based Manipulation on the Generative Image Manifold. [![stars](https://badgen.net/github/stars/XingangPan/DragGAN)](https://github.com/XingangPan/DragGAN)
  * [Segment Anything](https://github.com/facebookresearch/segment-anything): The repository provides code for running inference with the SegmentAnything Model (SAM), links for downloading the trained model checkpoints, and example notebooks that show how to use the model. [![stars](https://badgen.net/github/stars/facebookresearch/segment-anything)](https://github.com/facebookresearch/segment-anything)
  * [Segment Anything 2](https://github.com/facebookresearch/segment-anything-2): The repository provides code for running inference with the Meta Segment Anything Model 2 (SAM 2), links for downloading the trained model checkpoints, and example notebooks that show how to use the model. [![stars](https://badgen.net/github/stars/facebookresearch/segment-anything-2)](https://github.com/facebookresearch/segment-anything-2)
  * [MobileSAM](https://github.com/ChaoningZhang/MobileSAM): This is the official code for the MobileSAM project that makes SAM lightweight for mobile applications and beyond! [![stars](https://badgen.net/github/stars/ChaoningZhang/MobileSAM)](https://github.com/ChaoningZhang/MobileSAM)
  * [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM): Fast Segment Anything
 [![stars](https://badgen.net/github/stars/CASIA-IVA-Lab/FastSAM)](https://github.com/CASIA-IVA-Lab/FastSAM)
  * [Infinigen](https://github.com/princeton-vl/infinigen): Infinite Photorealistic Worlds using Procedural Generation. [![stars](https://badgen.net/github/stars/princeton-vl/infinigen)](https://github.com/princeton-vl/infinigen)
  * [DALL·E 3](https://openai.com/dall-e-3)
  * [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion): A Pipeline-Level Solution for Real-Time Interactive Generation. [![stars](https://badgen.net/github/stars/cumulo-autumn/StreamDiffusion)](https://github.com/cumulo-autumn/StreamDiffusion)
  * [AnyDoor](https://github.com/cumulo-autumn/StreamDiffusion): Zero-shot Object-level Image Customization. [![stars](https://badgen.net/github/stars/cumulo-autumn/StreamDiffusion)](https://github.com/cumulo-autumn/StreamDiffusion)
  * [DiT](https://github.com/facebookresearch/DiT): Scalable Diffusion Models with Transformers. [![stars](https://badgen.net/github/stars/facebookresearch/DiT)](https://github.com/facebookresearch/DiT)
  * [BrushNet](https://github.com/TencentARC/BrushNet): A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion. [![stars](https://badgen.net/github/stars/TencentARC/BrushNet)](https://github.com/TencentARC/BrushNet)
  * [OOTDiffusion](https://github.com/levihsu/OOTDiffusion): Outfitting Fusion based Latent Diffusion for Controllable Virtual Try-on. [![stars](https://badgen.net/github/stars/levihsu/OOTDiffusion)](https://github.com/levihsu/OOTDiffusion)
  * [VAR](https://github.com/FoundationVision/VAR): Official impl. of "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction". [![stars](https://badgen.net/github/stars/FoundationVision/VAR)](https://github.com/FoundationVision/VAR)
  * [Imagine Flash: Accelerating Emu Diffusion Models with Backward Distillation](https://ai.meta.com/research/publications/imagine-flash-accelerating-emu-diffusion-models-with-backward-distillation/)

##### 💡 Applications 💡

  * [ArtLine](https://github.com/vijishmadhavan/ArtLine): A Deep Learning based project for creating line art portraits. [![stars](https://badgen.net/github/stars/vijishmadhavan/ArtLine)](https://github.com/vijishmadhavan/ArtLine)
  * [Depix](https://github.com/beurtschipper/Depix): Recovers passwords from pixelized screenshots. [![stars](https://badgen.net/github/stars/beurtschipper/Depix)](https://github.com/beurtschipper/Depix)
  * [Bringing Old Photos Back to Life](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life): Old Photo Restoration (Official PyTorch Implementation). [![stars](https://badgen.net/github/stars/microsoft/Bringing-Old-Photos-Back-to-Life)](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life)
  * [Rewriting](https://github.com/davidbau/rewriting): Interactive tool to directly edit the rules of a GAN to synthesize scenes with objects added, removed, or altered. Change StyleGANv2 to make extravagant eyebrows, or horses wearing hats. [![stars](https://badgen.net/github/stars/davidbau/rewriting)](https://github.com/davidbau/rewriting)
  * [Fawkes](https://github.com/Shawn-Shan/fawkes): Privacy preserving tool against facial recognition systems. [![stars](https://badgen.net/github/stars/Shawn-Shan/fawkes)](https://github.com/Shawn-Shan/fawkes)
  * [Pulse](https://github.com/adamian98/pulse): Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models. [![stars](https://badgen.net/github/stars/adamian98/pulse)](https://github.com/adamian98/pulse)
  * [HiDT](https://github.com/saic-mdal/HiDT): Official repository for the paper "High-Resolution Daytime Translation Without Domain Labels". [![stars](https://badgen.net/github/stars/saic-mdal/HiDT)](https://github.com/saic-mdal/HiDT)
  * [3D Photo Inpainting](https://github.com/vt-vl-lab/3d-photo-inpainting): 3D Photography using Context-aware Layered Depth Inpainting. [![stars](https://badgen.net/github/stars/vt-vl-lab/3d-photo-inpainting)](https://github.com/vt-vl-lab/3d-photo-inpainting)
  * [SteganoGAN](https://github.com/DAI-Lab/SteganoGAN): SteganoGAN is a tool for creating steganographic images using adversarial training. [![stars](https://badgen.net/github/stars/DAI-Lab/SteganoGAN)](https://github.com/DAI-Lab/SteganoGAN)
  * [Stylegan-T](https://github.com/autonomousvision/stylegan-t): Unlocking the Power of GANs for Fast Large-Scale Text-to-Image Synthesis. [![stars](https://badgen.net/github/stars/autonomousvision/stylegan-t)](https://github.com/autonomousvision/stylegan-t)
  * [MegaPortraits](https://github.com/SamsungLabs/MegaPortraits): One-shot Megapixel Neural Head Avatars. [![stars](https://badgen.net/github/stars/SamsungLabs/MegaPortraits)](https://github.com/SamsungLabs/MegaPortraits)
  * [eg3d](https://github.com/NVlabs/eg3d): Efficient Geometry-aware 3D Generative Adversarial Networks. [![stars](https://badgen.net/github/stars/NVlabs/eg3d)](https://github.com/NVlabs/eg3d)
  * [TediGAN](https://github.com/IIGROUP/TediGAN): Pytorch implementation for TediGAN: Text-Guided Diverse Face Image Generation and Manipulation. [![stars](https://badgen.net/github/stars/IIGROUP/TediGAN)](https://github.com/IIGROUP/TediGAN)
  * [DALLE-pytorch](https://github.com/lucidrains/DALLE-pytorch): Implementation / replication of DALL-E, OpenAI's Text to Image Transformer, in Pytorch. [![stars](https://badgen.net/github/stars/lucidrains/DALLE-pytorch)](https://github.com/lucidrains/DALLE-pytorch)
  * [StyleNeRF](https://github.com/facebookresearch/StyleNeRF): This is the open source implementation of the ICLR2022 paper "StyleNeRF: A Style-based 3D-Aware Generator for High-resolution Image Synthesis". [![stars](https://badgen.net/github/stars/facebookresearch/StyleNeRF)](https://github.com/facebookresearch/StyleNeRF)
  * [DeepSVG](https://github.com/alexandre01/deepsvg): Official code for the paper "DeepSVG: A Hierarchical Generative Network for Vector Graphics Animation". Includes a PyTorch library for deep learning with SVG data. [![stars](https://badgen.net/github/stars/alexandre01/deepsvg)](https://github.com/alexandre01/deepsvg)
  * [NUWA](https://github.com/microsoft/NUWA): A unified 3D Transformer Pipeline for visual synthesis. [![stars](https://badgen.net/github/stars/microsoft/NUWA)](https://github.com/microsoft/NUWA)
  * [Image-Super-Resolution-via-Iterative-Refinement](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement): Unofficial implementation of Image Super-Resolution via Iterative Refinement by Pytorch. [![stars](https://badgen.net/github/stars/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement)
  * [Lama](https://github.com/saic-mdal/lama): 🦙 LaMa Image Inpainting, Resolution-robust Large Mask Inpainting with Fourier Convolutions. [![stars](https://badgen.net/github/stars/saic-mdal/lama)](https://github.com/saic-mdal/lama)
  * [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch): Pytorch ReID: A tiny, friendly, strong pytorch implement of object re-identification baseline. [![stars](https://badgen.net/github/stars/layumi/Person_reID_baseline_pytorch)](https://github.com/layumi/Person_reID_baseline_pytorch)
  * [instruct-pix2pix](https://github.com/timothybrooks/instruct-pix2pix): PyTorch implementation of InstructPix2Pix, an instruction-based image editing model. [![stars](https://badgen.net/github/stars/timothybrooks/instruct-pix2pix)](https://github.com/timothybrooks/instruct-pix2pix)
  * [GFPGAN](https://github.com/TencentARC/GFPGAN): GFPGAN aims at developing Practical Algorithms for Real-world Face Restoration. [![stars](https://badgen.net/github/stars/TencentARC/GFPGAN)](https://github.com/TencentARC/GFPGAN)
  * [DeepVecFont](https://github.com/yizhiwang96/deepvecfont): Synthesizing High-quality Vector Fonts via Dual-modality Learning. [![stars](https://badgen.net/github/stars/yizhiwang96/deepvecfont)](https://github.com/yizhiwang96/deepvecfont)
  * [Stargan v2 Tensorflow](https://github.com/clovaai/stargan-v2-tensorflow): Official Tensorflow Implementation. [![stars](https://badgen.net/github/stars/clovaai/stargan-v2-tensorflow)](https://github.com/clovaai/stargan-v2-tensorflow)
  * [StyleGAN2 Distillation](https://github.com/EvgenyKashin/stylegan2-distillation): Paired image-to-image translation, trained on synthetic data generated by StyleGAN2 outperforms existing approaches in image manipulation. [![stars](https://badgen.net/github/stars/EvgenyKashin/stylegan2-distillation)](https://github.com/EvgenyKashin/stylegan2-distillation)
  * [Extracting Training Data from Diffusion Models](https://arxiv.org/abs/2301.13188)
  * [Mann-E - Mann-E (Persian: مانی) is an art generator model based on the weights of Stable Diffusion 1.5 and data gathered from artistic material available on Pinterest](https://opencognitives.com/mann-e)
  * [End-to-end Trained CNN Encode-Decoder Networks for Image Steganography](https://arxiv.org/abs/1711.07201)
  * [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything): Marrying Grounding DINO with Segment Anything & Stable Diffusion & Tag2Text & BLIP & Whisper & ChatBot - Automatically Detect , Segment and Generate Anything with Image, Text, and Audio Inputs. [![stars](https://badgen.net/github/stars/IDEA-Research/Grounded-Segment-Anything)](https://github.com/IDEA-Research/Grounded-Segment-Anything)
  * [AnimateDiff](https://github.com/guoyww/AnimateDiff): Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning. [![stars](https://badgen.net/github/stars/guoyww/AnimateDiff)](https://github.com/guoyww/AnimateDiff)
  * [BasicSR](https://github.com/XPixelGroup/BasicSR): Open Source Image and Video Restoration Toolbox for Super-resolution, Denoise, Deblurring, etc. Currently, it includes EDSR, RCAN, SRResNet, SRGAN, ESRGAN, EDVR, BasicVSR, SwinIR, ECBSR, etc. Also support StyleGAN2 and DFDNet. [![stars](https://badgen.net/github/stars/XPixelGroup/BasicSR)](https://github.com/XPixelGroup/
  BasicSR)
  * [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN): Real-ESRGAN aims at developing Practical Algorithms for General Image/Video Restoration. [![stars](https://badgen.net/github/stars/xinntao/Real-ESRGAN)](https://github.com/xinntao/Real-ESRGAN)
  * [ESRGAN](https://github.com/xinntao/ESRGAN): Enhanced SRGAN. Champion PIRM Challenge on Perceptual Super-Resolution. [![stars](https://badgen.net/github/stars/xinntao/ESRGAN)](https://github.com/xinntao/ESRGAN)
  * [MixNMatch](https://github.com/Yuheng-Li/MixNMatch): Multifactor Disentanglement and Encoding for Conditional Image Generation. [![stars](https://badgen.net/github/stars/Yuheng-Li/MixNMatch)](https://github.com/Yuheng-Li/MixNMatch)
  * [Clarity-upscaler](https://github.com/philz1337x/clarity-upscaler): Reimagined image upscaling for everyone. [![stars](https://badgen.net/github/stars/philz1337x/clarity-upscaler)](https://github.com/philz1337x/clarity-upscaler)
  * [One-step Diffusion with Distribution Matching Distillation](https://tianweiy.github.io/dmd/)
  * [Invisible Stitch](https://github.com/paulengstler/invisible-stitch): Generating Smooth 3D Scenes with Depth Inpainting. [![stars](https://badgen.net/github/stars/paulengstler/invisible-stitch)](https://github.com/paulengstler/invisible-stitch)
  * [SSR](https://github.com/DaLi-Jack/SSR-code): Single-view 3D Scene Reconstruction with High-fidelity Shape and Texture. [![stars](https://badgen.net/github/stars/DaLi-Jack/SSR-code)](https://github.com/paulengstler/https://github.com/DaLi-Jack/SSR-code)
  * [InvSR](https://github.com/zsyOAOA/InvSR): Arbitrary-steps Image Super-resolution via Diffusion Inversion. [![stars](https://badgen.net/github/stars/zsyOAOA/InvSR)](https://github.com/zsyOAOA/InvSR)



##### 🔎 Detection 🔎

  * [stylegan3-detector](https://github.com/NVlabs/stylegan3-detector): StyleGAN3 Synthetic Image Detection. [![stars](https://badgen.net/github/stars/NVlabs/stylegan3-detector)](https://github.com/NVlabs/stylegan3-detector)
  * [stylegan2-projecting-images](https://github.com/woctezuma/stylegan2-projecting-images): Projecting images to latent space with StyleGAN2. [![stars](https://badgen.net/github/stars/woctezuma/stylegan2-projecting-images)](https://github.com/woctezuma/stylegan2-projecting-images)
  * [FALdetector](https://github.com/PeterWang512/FALdetector): Detecting Photoshopped Faces by Scripting Photoshop. [![stars](https://badgen.net/github/stars/PeterWang512/FALdetector)](https://github.com/PeterWang512/FALdetector)

#### 🎥 Video 🎥

##### 🛠️ Tools 🛠️

  * [DeepFaceLab](https://github.com/iperov/DeepFaceLab): DeepFaceLab is the leading software for creating deepfakes. [![stars](https://badgen.net/github/stars/iperov/DeepFaceLab)](https://github.com/iperov/DeepFaceLab)
  * [faceswap](https://github.com/deepfakes/faceswap): Deepfakes Software For All. [![stars](https://badgen.net/github/stars/deepfakes/faceswap)](https://github.com/deepfakes/faceswap)
  * [dot](https://github.com/sensity-ai/dot): The Deepfake Offensive Toolkit. [![stars](https://badgen.net/github/stars/sensity-ai/dot)](https://github.com/sensity-ai/dot)
  * [SimSwap](https://github.com/neuralchen/SimSwap): An arbitrary face-swapping framework on images and videos with one single trained model! [![stars](https://badgen.net/github/stars/neuralchen/SimSwap)](https://github.com/neuralchen/SimSwap)
  * [faceswap-GAN](https://github.com/shaoanlu/faceswap-GAN): A denoising autoencoder + adversarial losses and attention mechanisms for face swapping. [![stars](https://badgen.net/github/stars/shaoanlu/faceswap-GAN)](https://github.com/shaoanlu/faceswap-GAN)
  * [Celeb DeepFakeForensics](https://github.com/yuezunli/celeb-deepfakeforensics): A Large-scale Challenging Dataset for DeepFake Forensics. [![stars](https://badgen.net/github/stars/yuezunli/celeb-deepfakeforensics)](https://github.com/yuezunli/celeb-deepfakeforensics)
  * [VGen](https://github.com/damo-vilab/i2vgen-xl): A holistic video generation ecosystem for video generation building on diffusion models. [![stars](https://badgen.net/github/stars/damo-vilab/i2vgen-xl)](https://github.com/damo-vilab/i2vgen-xl)
  * [MuseV](https://github.com/TMElyralab/MuseV): Infinite-length and High Fidelity Virtual Human Video Generation with Visual Conditioned Parallel Denoising. [![stars](https://badgen.net/github/stars/TMElyralab/MuseV)](https://github.com/TMElyralab/MuseV)
  * [GLEE](https://github.com/FoundationVision/GLEE): General Object Foundation Model for Images and Videos at Scale. [![stars](https://badgen.net/github/stars/FoundationVision/GLEE)](https://github.com/FoundationVision/GLEE)
  * [T-Rex](https://github.com/IDEA-Research/T-Rex): Towards Generic Object Detection via Text-Visual Prompt Synergy. [![stars](https://badgen.net/github/stars/IDEA-Research/T-Rex)](https://github.com/IDEA-Research/T-Rex)
  * [DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter): Animating Open-domain Images with Video Diffusion Priors. [![stars](https://badgen.net/github/stars/Doubiiu/DynamiCrafter)](https://github.com/Doubiiu/DynamiCrafter)
  * [Mora](https://github.com/lichao-sun/Mora): More like Sora for Generalist Video Generation. [![stars](https://badgen.net/github/stars/lichao-sun/Mora)](https://github.com/lichao-sun/Mora)

##### 💡 Applications 💡

  * [face2face-demo](https://github.com/datitran/face2face-demo): pix2pix demo that learns from facial landmarks and translates this into a face. [![stars](https://badgen.net/github/stars/datitran/face2face-demo)](https://github.com/datitran/face2face-demo)
  * [Faceswap-Deepfake-Pytorch](https://github.com/Oldpan/Faceswap-Deepfake-Pytorch): Faceswap with Pytorch or DeepFake with Pytorch. [![stars](https://badgen.net/github/stars/Oldpan/Faceswap-Deepfake-Pytorch)](https://github.com/Oldpan/Faceswap-Deepfake-Pytorch)
  * [Point-E](https://github.com/openai/point-e): Point cloud diffusion for 3D model synthesis. [![stars](https://badgen.net/github/stars/openai/point-e)](https://github.com/openai/point-e)
  * [EGVSR](https://github.com/Thmen/EGVSR): Efficient & Generic Video Super-Resolution. [![stars](https://badgen.net/github/stars/Thmen/EGVSR)](https://github.com/Thmen/EGVSR)
  * [STIT](https://github.com/rotemtzaban/STIT): Stitch it in Time: GAN-Based Facial Editing of Real Videos. [![stars](https://badgen.net/github/stars/rotemtzaban/STIT)](https://github.com/rotemtzaban/STIT)
  * [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2): Real-Time High-Resolution Background Matting. [![stars](https://badgen.net/github/stars/PeterL1n/BackgroundMattingV2)](https://github.com/PeterL1n/BackgroundMattingV2)
  * [MODNet](https://github.com/ZHKKKe/MODNet): A Trimap-Free Portrait Matting Solution in Real Time. [![stars](https://badgen.net/github/stars/ZHKKKe/MODNet)](https://github.com/ZHKKKe/MODNet)
  * [Background-Matting](https://github.com/senguptaumd/Background-Matting): Background Matting: The World is Your Green Screen. [![stars](https://badgen.net/github/stars/senguptaumd/Background-Matting)](https://github.com/senguptaumd/Background-Matting)
  * [First Order Model](https://github.com/AliaksandrSiarohin/first-order-model): This repository contains the source code for the paper First Order Motion Model for Image Animation. [![stars](https://badgen.net/github/stars/AliaksandrSiarohin/first-order-model)](https://github.com/AliaksandrSiarohin/first-order-model)
  * [Articulated Animation](https://github.com/snap-research/articulated-animation): This repository contains the source code for the CVPR'2021 paper Motion Representations for Articulated Animation. [![stars](https://badgen.net/github/stars/snap-research/articulated-animation)](https://github.com/snap-research/articulated-animation)
  * [Real Time Person Removal](https://github.com/jasonmayes/Real-Time-Person-Removal): Removing people from complex backgrounds in real time using TensorFlow.js in the web browser. [![stars](https://badgen.net/github/stars/jasonmayes/Real-Time-Person-Removal)](https://github.com/jasonmayes/Real-Time-Person-Removal)
  * [AdaIN-style](https://github.com/xunhuang1995/AdaIN-style): Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization. [![stars](https://badgen.net/github/stars/xunhuang1995/AdaIN-style)](https://github.com/xunhuang1995/AdaIN-style)
  * [Frame Interpolation](https://github.com/google-research/frame-interpolation): Frame Interpolation for Large Motion. [![stars](https://badgen.net/github/stars/google-research/frame-interpolation)](https://github.com/google-research/frame-interpolation)
  * [Awesome-Image-Colorization](https://github.com/MarkMoHR/Awesome-Image-Colorization): 📚 A collection of Deep Learning based Image Colorization and Video Colorization papers. [![stars](https://badgen.net/github/stars/MarkMoHR/Awesome-Image-Colorization)](https://github.com/MarkMoHR/Awesome-Image-Colorization)
  * [SadTalker](https://github.com/OpenTalker/SadTalker): Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation. [![stars](https://badgen.net/github/stars/OpenTalker/SadTalker)](https://github.com/OpenTalker/SadTalker)
  * [roop](https://github.com/s0md3v/roop): One-click deepfake (face swap). [![stars](https://badgen.net/github/stars/s0md3v/roop)](https://github.com/s0md3v/roop)
  * [StableVideo](https://github.com/rese1f/StableVideo): Text-driven Consistency-aware Diffusion Video Editing. [![stars](https://badgen.net/github/stars/rese1f/StableVideo)](https://github.com/rese1f/StableVideo)
  * [MagicEdit](https://github.com/magic-research/magic-edit): High-Fidelity Temporally Coherent Video Editing. [![stars](https://badgen.net/github/stars/magic-research/magic-edit)](https://github.com/magic-research/magic-edit)
  * [Rerender_A_Video](https://github.com/williamyang1991/Rerender_A_Video): Zero-Shot Text-Guided Video-to-Video Translation. [![stars](https://badgen.net/github/stars/williamyang1991/Rerender_A_Video)](https://github.com/williamyang1991/Rerender_A_Video)
  * [DreamEditor](https://github.com/zjy526223908/DreamEditor): Text-Driven 3D Scene Editing with Neural Fields. [![stars](https://badgen.net/github/stars/zjy526223908/DreamEditor)](https://github.com/zjy526223908/DreamEditor)
  * [DreamEditor](https://github.com/zju3dv/4K4D): Real-Time 4D View Synthesis at 4K Resolution. [![stars](https://badgen.net/github/stars/zju3dv/4K4D)](https://github.com/zju3dv/4K4D)
  * [AnimateAnyone](https://github.com/HumanAIGC/AnimateAnyone): Consistent and Controllable Image-to-Video Synthesis for Character Animation. [![stars](https://badgen.net/github/stars/HumanAIGC/AnimateAnyone)](https://github.com/HumanAIGC/AnimateAnyone)
  * [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone): This repository reproduces AnimateAnyone. [![stars](https://badgen.net/github/stars/MooreThreads/Moore-AnimateAnyone)](https://github.com/MooreThreads/Moore-AnimateAnyone)
  * [audio2photoreal](https://github.com/facebookresearch/audio2photoreal): From Audio to Photoreal Embodiment: Synthesizing Humans in Conversations. [![stars](https://badgen.net/github/stars/facebookresearch/audio2photoreal)](https://github.com/facebookresearch/audio2photoreal)
  * [MagicVideo-V2: Multi-Stage High-Aesthetic Video Generation](https://magicvideov2.github.io/)
  * [LWM](https://github.com/LargeWorldModel/LWM): A general-purpose large-context multimodal autoregressive model. It is trained on a large dataset of diverse long videos and books using RingAttention and can perform language, image, and video understanding and generation. [![stars](https://badgen.net/github/stars/LargeWorldModel/LWM)](https://github.com/LargeWorldModel/LWM)
  * [AniPortrait](https://github.com/Zejun-Yang/AniPortrait): Audio-Driven Synthesis of Photorealistic Portrait Animation. [![stars](https://badgen.net/github/stars/Zejun-Yang/AniPortrait)](https://github.com/Zejun-Yang/AniPortrait)
  * [Champ](https://github.com/fudan-generative-vision/champ): Controllable and Consistent Human Image Animation with 3D Parametric Guidance. [![stars](https://badgen.net/github/stars/fudan-generative-vision/champ)](https://github.com/fudan-generative-vision/champ)
  * [Streamv2v](https://github.com/Jeff-LiangF/streamv2v): Streaming Video-to-Video Translation with Feature Banks. [![stars](https://badgen.net/github/stars/Jeff-LiangF/streamv2v)](https://github.com/Jeff-LiangF/streamv2v)
  * [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam): Real time face swap and one-click video deepfake with only a single image. [![stars](https://badgen.net/github/stars/hacksider/Deep-Live-Cam)](https://github.com/hacksider/Deep-Live-Cam)
  * [Sapiens](https://github.com/facebookresearch/sapiens): Foundation for Human Vision Models. [![stars](https://badgen.net/github/stars/facebookresearch/sapiens)](https://github.com/facebookresearch/sapiens)


##### 🔎 Detection 🔎

  * [FaceForensics++](https://github.com/ondyari/FaceForensics): FaceForensics dataset. [![stars](https://badgen.net/github/stars/ondyari/FaceForensics)](https://github.com/ondyari/FaceForensics)
  * [DeepFake-Detection](https://github.com/dessa-oss/DeepFake-Detection): Towards deepfake detection that actually works. [![stars](https://badgen.net/github/stars/dessa-oss/DeepFake-Detection)](https://github.com/dessa-oss/DeepFake-Detection)
  * [fakeVideoForensics](https://github.com/jiep/fakeVideoForensics): Detect deep fakes videos. [![stars](https://badgen.net/github/stars/jiep/fakeVideoForensics)](https://github.com/jiep/fakeVideoForensics)
  * [Deepfake-Detection](https://github.com/HongguLiu/Deepfake-Detection): The Pytorch implemention of Deepfake Detection based on Faceforensics++. [![stars](https://badgen.net/github/stars/HongguLiu/Deepfake-Detection)](https://github.com/HongguLiu/Deepfake-Detection)
  * [SeqDeepFake](https://github.com/rshaojimmy/SeqDeepFake): PyTorch code for SeqDeepFake: Detecting and Recovering Sequential DeepFake Manipulation. [![stars](https://badgen.net/github/stars/rshaojimmy/SeqDeepFake)](https://github.com/rshaojimmy/SeqDeepFake)
  * [PCL-I2G](https://github.com/jtchen0528/PCL-I2G): Unofficial Implementation: Learning Self-Consistency for Deepfake Detection. [![stars](https://badgen.net/github/stars/jtchen0528/PCL-I2G)](https://github.com/jtchen0528/PCL-I2G)
  * [DFDC DeepFake Challenge](https://github.com/selimsef/dfdc_deepfake_challenge): A prize winning solution for DFDC challenge. [![stars](https://badgen.net/github/stars/selimsef/dfdc_deepfake_challenge)](https://github.com/selimsef/dfdc_deepfake_challenge)
  * [POI-Forensics](https://github.com/grip-unina/poi-forensics): Audio-Visual Person-of-Interest DeepFake Detection. [![stars](https://badgen.net/github/stars/grip-unina/poi-forensics)](https://github.com/grip-unina/poi-forensics)
  * [Standardizing Detection of Deepfakes: Why Experts Say It’s Important](https://antispoofing.org/deepfake-detection-standardization-origin-goals-and-implementation/)
  * [Want to spot a deepfake? Look for the stars in their eyes](https://ras.ac.uk/news-and-press/news/want-spot-deepfake-look-stars-their-eyes)

#### 📄 Text 📄

##### 🛠️ Tools 🛠️
  * [GLM-130B](https://github.com/THUDM/GLM-130B): An Open Bilingual Pre-Trained Model. [![stars](https://badgen.net/github/stars/THUDM/GLM-130B)](https://github.com/THUDM/GLM-130B)
  * [LongtermChatExternalSources](https://github.com/daveshap/LongtermChatExternalSources): GPT-3 chatbot with long-term memory and external sources. [![stars](https://badgen.net/github/stars/daveshap/LongtermChatExternalSources)](https://github.com/daveshap/LongtermChatExternalSources)
  * [sketch](https://github.com/approximatelabs/sketch): AI code-writing assistant that understands data content. [![stars](https://badgen.net/github/stars/approximatelabs/sketch)](https://github.com/approximatelabs/sketch)
  * [LangChain](https://github.com/hwchase17/langchain): ⚡ Building applications with LLMs through composability ⚡. [![stars](https://badgen.net/github/stars/hwchase17/langchain)](https://github.com/hwchase17/langchain)
  * [ChatGPT Wrapper](https://github.com/mmabrouk/chatgpt-wrapper): API for interacting with ChatGPT using Python and from Shell. [![stars](https://badgen.net/github/stars/mmabrouk/chatgpt-wrapper)](https://github.com/mmabrouk/chatgpt-wrapper)
  * [openai-python](https://github.com/openai/openai-python): The OpenAI Python library provides convenient access to the OpenAI API from applications written in the Python language. [![stars](https://badgen.net/github/stars/openai/openai-python)](https://github.com/openai/openai-python)
  * [Beto](https://github.com/dccuchile/beto): Spanish version of the BERT model. [![stars](https://badgen.net/github/stars/dccuchile/beto)](https://github.com/dccuchile/beto)
  * [GPT-Code-Clippy](https://github.com/CodedotAl/gpt-code-clippy): GPT-Code-Clippy (GPT-CC) is an open source version of GitHub Copilot, a language model -- based on GPT-3, called GPT-Codex. [![stars](https://badgen.net/github/stars/CodedotAl/gpt-code-clippy)](https://github.com/CodedotAl/gpt-code-clippy)
  * [GPT Neo](https://github.com/EleutherAI/gpt-neo): An implementation of model parallel GPT-2 and GPT-3-style models using the mesh-tensorflow library. [![stars](https://badgen.net/github/stars/EleutherAI/gpt-neo)](https://github.com/EleutherAI/gpt-neo)
  * [ctrl](https://github.com/salesforce/ctrl): Conditional Transformer Language Model for Controllable Generation. [![stars](https://badgen.net/github/stars/salesforce/ctrl)](https://github.com/salesforce/ctrl)
  * [Llama](https://github.com/facebookresearch/llama): Inference code for LLaMA models. [![stars](https://badgen.net/github/stars/facebookresearch/llama)](https://github.com/facebookresearch/llama)
  * [Llama2](https://ai.meta.com/llama/)
  * [Llama Guard 3](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama-guard-3/)
  * [UL2 20B](https://ai.googleblog.com/2022/10/ul2-20b-open-source-unified-language.html): An Open Source Unified Language Learner
  * [burgpt](https://github.com/aress31/burpgpt): A Burp Suite extension that integrates OpenAI's GPT to perform an additional passive scan for discovering highly bespoke vulnerabilities, and enables running traffic-based analysis of any type. [![stars](https://badgen.net/github/stars/aress31/burpgpt)](https://github.com/aress31/burpgpt)
  * [Ollama](https://github.com/jmorganca/ollama): Get up and running with Llama 2 and other large language models locally. [![stars](https://badgen.net/github/stars/jmorganca/ollama)](https://github.com/jmorganca/ollama)
  * [SneakyPrompt](https://github.com/Yuchen413/text2image_safety): Jailbreaking Text-to-image Generative Models. [![stars](https://badgen.net/github/stars/Yuchen413/text2image_safety)](https://github.com/Yuchen413/text2image_safety)
    * [Copilot-For-Security](https://github.com/Azure/Copilot-For-Security): A generative AI-powered security solution that helps increase the efficiency and capabilities of defenders to improve security outcomes at machine speed and scale, while remaining compliant with responsible AI principles. [![stars](https://badgen.net/github/stars/Azure/Copilot-For-Security)](https://github.com/Azure/Copilot-For-Security)
  * [LM Studio](https://lmstudio.ai/): Discover, download, and run local LLMs
  * [Bypass GPT: Convert AI Text to Human-like Content](https://bypassgpt.ai/)
  * [MGM](https://github.com/dvlab-research/MGM): The framework supports a series of dense and MoE Large Language Models (LLMs) from 2B to 34B with image understanding, reasoning, and generation simultaneously. [![stars](https://badgen.net/github/stars/dvlab-research/MGM)](https://github.com/dvlab-research/MGM)
  * [Secret Llama](https://github.com/abi/secret-llama): Fully private LLM chatbot that runs entirely with a browser with no server needed. Supports Mistral and LLama 3. [![stars](https://badgen.net/github/stars/abi/secret-llama)](https://github.com/abi/secret-llama)
  * [Llama3](https://github.com/meta-llama/llama3): The official Meta Llama 3 GitHub site. [![stars](https://badgen.net/github/stars/meta-llama/llama3)](https://github.com/meta-llama/llama3)

##### 🔎 Detection 🔎

  * [Detecting Fake Text](https://github.com/HendrikStrobelt/detecting-fake-text): Giant Language Model Test Room. [![stars](https://badgen.net/github/stars/HendrikStrobelt/detecting-fake-text)](https://github.com/HendrikStrobelt/detecting-fake-text)
  * [Grover](https://github.com/rowanz/grover): Code for Defending Against Neural Fake News. [![stars](https://badgen.net/github/stars/rowanz/grover)](https://github.com/rowanz/grover)
  * [Rebuff.ai](https://github.com/protectai/rebuff):  Prompt Injection Detector. [![stars](https://badgen.net/github/stars/protectai/rebuff)](https://github.com/protectai/rebuff)
  * [New AI classifier for indicating AI-written text](https://openai.com/blog/new-ai-classifier-for-indicating-ai-written-text/)
  * [Discover the 4 Magical Methods to Detect AI-Generated Text (including ChatGPT)](https://medium.com/@itamargolan/uncover-the-four-enchanted-ways-to-identify-ai-generated-text-including-chatgpts-4764847fd609)
  * [GPTZero](https://gptzero.me)
  * [AI Content Detector (beta)](https://copyleaks.com/ai-content-detector)
  * [A Watermark for Large Language Models](https://arxiv.org/abs/2301.10226)
  * [Can AI-Generated Text be Reliably Detected?](https://arxiv.org/abs/2303.11156)
  * [GPT detectors are biased against non-native English writers](https://arxiv.org/abs/2304.02819)
  * [To ChatGPT, or not to ChatGPT: That is the question!](https://arxiv.org/abs/2304.01487)
  * [Can linguists distinguish between ChatGPT/AI and human writing?: A study of research ethics and academic publishing](https://www.sciencedirect.com/science/article/abs/pii/S2772766123000289)
  * [ChatGPT is bullshit](https://link.springer.com/article/10.1007/s10676-024-09775-5)


##### 💡 Applications 💡

  * [handwrite](https://github.com/builtree/handwrite): Handwrite generates a custom font based on your handwriting sample. [![stars](https://badgen.net/github/stars/builtree/handwrite)](https://github.com/builtree/handwrite)
  * [GPT Sandbox](https://github.com/shreyashankar/gpt3-sandbox): The goal of this project is to enable users to create cool web demos using the newly released OpenAI GPT-3 API with just a few lines of Python. [![stars](https://badgen.net/github/stars/shreyashankar/gpt3-sandbox)](https://github.com/shreyashankar/gpt3-sandbox)
  * [PassGAN](https://github.com/brannondorsey/PassGAN): A Deep Learning Approach for Password Guessing. [![stars](https://badgen.net/github/stars/brannondorsey/PassGAN)](https://github.com/brannondorsey/PassGAN)
  * [GPT Index](https://github.com/jerryjliu/gpt_index): GPT Index is a project consisting of a set of data structures designed to make it easier to use large external knowledge bases with LLMs. [![stars](https://badgen.net/github/stars/jerryjliu/gpt_index)](https://github.com/jerryjliu/gpt_index)
  * [nanoGPT](https://github.com/karpathy/nanoGPT): The simplest, fastest repository for training/finetuning medium-sized GPTs. [![stars](https://badgen.net/github/stars/karpathy/nanoGPT)](https://github.com/karpathy/nanoGPT)
  * [whatsapp-gpt](https://github.com/danielgross/whatsapp-gpt) [![stars](https://badgen.net/github/stars/danielgross/whatsapp-gpt)](https://github.com/danielgross/whatsapp-gpt)
  * [ChatGPT Chrome Extension](https://github.com/gragland/chatgpt-chrome-extension): A ChatGPT Chrome extension. Integrates ChatGPT into every text box on the internet.
  * [Unilm](https://github.com/microsoft/unilm): Large-scale Self-supervised Pre-training Across Tasks, Languages, and Modalities. [![stars](https://badgen.net/github/stars/microsoft/unilm)](https://github.com/microsoft/unilm)
  * [minGPT](https://github.com/karpathy/minGPT): A minimal PyTorch re-implementation of the OpenAI GPT (Generative Pretrained Transformer) training. [![stars](https://badgen.net/github/stars/karpathy/minGPT)](https://github.com/karpathy/minGPT)
  * [CodeGeeX](https://github.com/THUDM/CodeGeeX): An Open Multilingual Code Generation Model. [![stars](https://badgen.net/github/stars/THUDM/CodeGeeX)](https://github.com/THUDM/CodeGeeX)
  * [OpenAI Cookbook](https://github.com/openai/openai-cookbook): Examples and guides for using the OpenAI API. [![stars](https://badgen.net/github/stars/openai/openai-cookbook)](https://github.com/openai/openai-cookbook)
  * [🧠 Awesome ChatGPT Prompts](https://github.com/f/awesome-chatgpt-prompts): This repo includes ChatGPT prompt curation to use ChatGPT better. [![stars](https://badgen.net/github/stars/f/awesome-chatgpt-prompts)](https://github.com/f/awesome-chatgpt-prompts)
  * [Alice](https://github.com/greshake/Alice): Giving ChatGPT access to a real terminal. [![stars](https://badgen.net/github/stars/greshake/Alice)](https://github.com/greshake/Alice)
  * [Security Code Review With ChatGPT](https://research.nccgroup.com/2023/02/09/security-code-review-with-chatgpt)
  * [Do Users Write More Insecure Code with AI Assistants?](https://arxiv.org/abs/2211.03622)
  * [Bypassing Gmail's spam filters with ChatGPT](https://neelc.org/posts/chatgpt-gmail-spam)
  * [Recurrent GANs Password Cracker For IoT Password Security Enhancement](https://www.mdpi.com/1999-4893/15/5/155)
  * [PentestGPT](https://github.com/GreyDGL/PentestGPT): A GPT-empowered penetration testing tool. [![stars](https://badgen.net/github/stars/GreyDGL/PentestGPT)](https://github.com/GreyDGL/PentestGPT)
  * [GPT Researcher](https://github.com/assafelovic/gpt-researcher): GPT based autonomous agent that does online comprehensive research on any given topic. [![stars](https://badgen.net/github/stars/assafelovic/gpt-researcher)](https://github.com/assafelovic/gpt-researcher)
  * [GPT Engineer](https://github.com/AntonOsika/gpt-engineer): Specify what you want it to build, the AI asks for clarification, and then builds it. [![stars](https://badgen.net/github/stars/AntonOsika/gpt-engineer)](https://github.com/AntonOsika/gpt-engineer)
  * [localpilot](https://github.com/danielgross/localpilot): Use GitHub Copilot locally on your Macbook with one-click! [![stars](https://badgen.net/github/stars/danielgross/localpilot)](https://github.com/danielgross/localpilot)
  * [WormGPT])(https://thehackernews.com/2023/07/wormgpt-new-ai-tool-allows.html): New AI Tool Allows Cybercriminals to Launch Sophisticated Cyber Attacks
  * [PoisonGPT](https://blog.mithrilsecurity.io/poisongpt-how-we-hid-a-lobotomized-llm-on-hugging-face-to-spread-fake-news/): How we hid a lobotomized LLM on Hugging Face to spread fake news
  * [PassGPT: Password Modeling and (Guided) Generation with Large Language Models](https://javirandor.github.io/assets/pdf/passgpt2023rando.pdf)
  * [DeepPass — Finding Passwords With Deep Learning](https://posts.specterops.io/deeppass-finding-passwords-with-deep-learning-4d31c534cd00)
  * [GPTFuzz](https://github.com/sherdencooper/GPTFuzz): Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts. [![stars](https://badgen.net/github/stars/sherdencooper/GPTFuzz)](https://github.com/sherdencooper/GPTFuzz)
  * [Open Interpreter](https://github.com/KillianLucas/open-interpreter): OpenAI's Code Interpreter in your terminal, running locally. [![stars](https://badgen.net/github/stars/KillianLucas/open-interpreter)](https://github.com/KillianLucas/open-interpreter)
  * [Eureka](https://github.com/eureka-research/Eureka): Human-Level Reward Design via Coding Large Language Models. [![stars](https://badgen.net/github/stars/eureka-research/Eureka)](https://github.com/eureka-research/Eureka)
  * [MetaCLIP](https://github.com/facebookresearch/MetaCLIP): Demystifying CLIP Data. [![stars](https://badgen.net/github/stars/facebookresearch/MetaCLIP)](https://github.com/facebookresearch/MetaCLIP)
  * [LLM OSINT](https://github.com/sshh12/llm_osint): Proof-of-concept method of using LLMs to gather information from the internet and then perform a task with this information. [![stars](https://badgen.net/github/stars/sshh12/llm_osint)](https://github.com/sshh12/llm_osint)
  * [HackingBuddyGPT](https://github.com/ipa-lab/hackingBuddyGPT): LLMs x PenTesting. [![stars](https://badgen.net/github/stars/ipa-lab/hackingBuddyGPT)](https://github.com/ipa-lab/hackingBuddyGPT)
  * [ChatGPT-Jailbreaks](https://github.com/GabryB03/ChatGPT-Jailbreaks): Official jailbreak for ChatGPT (GPT-3.5). Send a long message at the start of the conversation with ChatGPT to get offensive, unethical, aggressive, human-like answers in English and Italian. [![stars](https://badgen.net/github/stars/GabryB03/ChatGPT-Jailbreaks)](https://github.com/GabryB03/ChatGPT-Jailbreaks)
  * [Magika](https://github.com/google/magika): Detect file content types with deep learning. [![stars](https://badgen.net/github/stars/google/magika)](https://github.com/google/magika)
  * [Jan](https://github.com/janhq/jan): An open source alternative to ChatGPT that runs 100% offline on your computer. [![stars](https://badgen.net/github/stars/janhq/jan)](https://github.com/janhq/jan)
  * [LibreChat](https://github.com/danny-avila/LibreChat): Enhanced ChatGPT Clone: Features OpenAI, Assistants API, Azure, Groq, GPT-4 Vision, Mistral, Bing, Anthropic, OpenRouter, Vertex AI, Gemini, AI model switching, message search, langchain, DALL-E-3, ChatGPT Plugins, OpenAI Functions, Secure Multi-User System, Presets, completely open-source for self-hosting. [![stars](https://badgen.net/github/stars/danny-avila/LibreChat)](https://github.com/danny-avila/LibreChat)
  * [Lumina-T2X](https://github.com/Alpha-VLLM/Lumina-T2X): A unified framework for Text to Any Modality Generation. [![stars](https://badgen.net/github/stars/Alpha-VLLM/Lumina-T2X)](https://github.com/Alpha-VLLM/Lumina-T2X)


### 📚 Misc 📚

  * [Awesome GPT + Security](https://github.com/cckuailong/awesome-gpt-security): A curated list of awesome security tools, experimental case or other interesting things with LLM or GPT. [![stars](https://badgen.net/github/stars/cckuailong/awesome-gpt-security)](https://github.com/cckuailong/awesome-gpt-security)
  * [🚀 Awesome Reinforcement Learning for Cyber Security](https://github.com/Limmen/awesome-rl-for-cybersecurity): A curated list of resources dedicated to reinforcement learning applied to cyber security. [![stars](https://badgen.net/github/stars/Limmen/awesome-rl-for-cybersecurity)](https://github.com/Limmen/awesome-rl-for-cybersecurity)
  * [Awesome Machine Learning for Cyber Security](https://github.com/jivoi/awesome-ml-for-cybersecurity): A curated list of amazingly awesome tools and resources related to the use of machine learning for cyber security. [![stars](https://badgen.net/github/stars/jivoi/awesome-ml-for-cybersecurity)](https://github.com/jivoi/awesome-ml-for-cybersecurity)
  * [Hugging Face Diffusion Models Course](https://github.com/huggingface/diffusion-models-class): Materials for the Hugging Face Diffusion Models Course. [![stars](https://badgen.net/github/stars/huggingface/diffusion-models-class)](https://github.com/huggingface/diffusion-models-class)
  * [Awesome-AI-Security](https://github.com/DeepSpaceHarbor/Awesome-AI-Security): A curated list of AI security resources. [![stars](https://badgen.net/github/stars/DeepSpaceHarbor/Awesome-AI-Security)](https://github.com/DeepSpaceHarbor/Awesome-AI-Security)
  * [ML for Hackers](https://github.com/johnmyleswhite/ML_for_Hackers): Code accompanying the book "Machine Learning for Hackers". [![stars](https://badgen.net/github/stars/johnmyleswhite/ML_for_Hackers)](https://github.com/johnmyleswhite/ML_for_Hackers)
  * [Awful AI](https://github.com/daviddao/awful-ai): Awful AI is a curated list to track current scary usages of AI - hoping to raise awareness. [![stars](https://badgen.net/github/stars/daviddao/awful-ai)](https://github.com/daviddao/awful-ai)
  * [NIST AI Risk Management Framework Playbook](https://pages.nist.gov/AIRMF)
  * [SoK: Explainable Machine Learning for Computer Security Applications](https://arxiv.org/abs/2208.10605)
  * [Who Evaluates the Evaluators? On Automatic Metrics for Assessing AI-based Offensive Code Generators](https://arxiv.org/abs/2212.06008)
  * [Vulnerability Prioritization: An Offensive Security Approach](https://arxiv.org/abs/2206.11182)
  * [MITRE ATLAS™](https://atlas.mitre.org) (Adversarial Threat Landscape for Artificial-Intelligence Systems)
  * [A Survey on Reinforcement Learning Security with Application to Autonomous Driving](https://arxiv.org/abs/2212.06123)
  * [How to avoid machine learning pitfalls: a guide for academic researchers](https://arxiv.org/abs/2108.02497)
  * [A curated list of AI Security & Privacy events](https://github.com/ZhengyuZhao/AI-Security-and-Privacy-Events)
  * [NIST AI 100-2 E2023](https://csrc.nist.gov/pubs/ai/100/2/e2023/final): Adversarial Machine Learning. A Taxonomy and Terminology of Attacks and Mitigations.
  * [🇪🇸 RootedCon 2023 - Inteligencia artificial ofensiva - ¿Cómo podemos estar  preparados?](/slides/RootedCon_2023.pdf)
  * [Security of AI-Systems: Fundamentals - Adversarial Deep Learning](https://www.bsi.bund.de/SharedDocs/Downloads/EN/BSI/KI/Security-of-AI-systems_fundamentals.pdf)
  * [Beyond the Safeguards: Exploring the Security Risks of ChatGPT](https://arxiv.org/abs/2305.08005)
  * [The AI Attack Surface Map v1.0](https://danielmiessler.com/blog/the-ai-attack-surface-map-v1-0)
  * [On the Impossible Safety of Large AI Models](https://arxiv.org/abs/2209.15259)
  * [Frontier AI Regulation: Managing Emerging Risks to Public Safety](https://arxiv.org/abs/2307.03718)
  * [Multilayer Framework for Good Cybersecurity Practices for AI](https://www.enisa.europa.eu/publications/multilayer-framework-for-good-cybersecurity-practices-for-ai)
  * [Introducing Google’s Secure AI Framework](https://blog.google/technology/safety-security/introducing-googles-secure-ai-framework/)
  * [OWASP Top 10 for LLM](https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-2023-v1_0.pdf)
  * [Awesome LLM Security](https://github.com/corca-ai/awesome-llm-security): A curation of awesome tools, documents and projects about LLM Security. [![stars](https://badgen.net/github/stars/corca-ai/awesome-llm-security)](https://github.com/corca-ai/awesome-llm-security)
  * A framework to securely use LLMs in companies. [Part 1: Overview of Risks](https://boringappsec.substack.com/p/edition-21-a-framework-to-securely). [Part 2: Managing risk](https://boringappsec.substack.com/p/edition-22-a-framework-to-securely). [Part 3: Securing ChatGPT and GitHub Copilot](https://boringappsec.substack.com/p/edition-23-a-framework-to-securely).
  * [A Study on Robustness and Reliability of Large Language Model Code Generation](https://arxiv.org/abs/2308.10335)
  * [Identifying AI-generated images with SynthID](https://www.deepmind.com/blog/identifying-ai-generated-images-with-synthid)
  * [Auditing Large Language Models: A Three-Layered Approach](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4361607)
  * [Resolving the battle of short‑ vs. long‑term AI risk](https://link.springer.com/content/pdf/10.1007/s43681-023-00336-y.pdf)
  * [FraudGPT: The Villain Avatar of ChatGPT](https://netenrich.com/blog/fraudgpt-the-villain-avatar-of-chatgpt)
  * [AI Risks - Schneier on Security](https://www.schneier.com/blog/archives/2023/10/ai-risks.html)
  * [Use of LLMs for Illicit Purposes: Threats, Prevention Measures, and Vulnerabilities](https://arxiv.org/abs/2308.12833)
  * [AI Red-Teaming Is Not a One-Stop Solution to AI Harms: Recommendations for Using Red-Teaming for AI Accountability](https://datasociety.net/wp-content/uploads/2023/10/Recommendations-for-Using-Red-Teaming-for-AI-Accountability-PolicyBrief.pdf)
  * [A Taxonomy of Trustworthiness for Artificial Intelligence](https://cltc.berkeley.edu/wp-content/uploads/2023/01/Taxonomy_of_AI_Trustworthiness.pdf)
  * [Managing AI Risks in an Era of Rapid Progress](https://arxiv.org/abs/2310.17688)
  * [Google - Acting on our commitment to safe and secure AI](https://blog.google/technology/safety-security/google-ai-security-expansion/)
  * [Offensive ML Playbook](https://wiki.offsecml.com/Welcome+to+the+Offensive+ML+Playbook)
  * [Demystifying Generative AI 🤖 A Security Researcher's Notes](https://blog.openthreatresearch.com/demystifying-generative-ai-a-security-researchers-notes/)
  * [GenAI-Security-Adventures](https://github.com/OTRF/GenAI-Security-Adventures): An open-source initiative to share notes, presentations, and a diverse collection of experiments presented in Jupyter Notebooks, all aimed at helping you grasp the essential concepts behind large language models and exploring the intriguing intersection of security and natural language processing. [![stars](https://badgen.net/github/stars/OTRF/GenAI-Security-Adventures)](https://github.com/OTRF/GenAI-Security-Adventures)
  * [AI Safety Camp](https://aisafety.camp/) connects you with a research lead to collaborate on a project – to see where your work could help ensure future AI is safe.
  * [Guidelines for secure AI system development](https://www.ncsc.gov.uk/files/Guidelines-for-secure-AI-system-development.pdf)
  * [Approach to Artificial Intelligence and Cybersecurity. BEST PRACTICE REPORT](https://www.ccn-cert.cni.es/es/informes/informes-de-buenas-practicas-bp/7192-ccn-cert-bp-30-approach-to-artificial-intelligence-and-cybersecurity/file.html)
  * [Stanford Safe, Secure, and Trustworthy AI EO 14110 Tracker](https://docs.google.com/spreadsheets/d/1xOL4hkQ2pLR-IAs3awIiXjPLmhIeXyE5-giJ5nT-h1M/edit#gid=142633882)
  * [Awesome ML Security](https://github.com/trailofbits/awesome-ml-security): A curated list of awesome machine learning security references, guidance, tools, and more. [![stars](https://badgen.net/github/stars/trailofbits/awesome-ml-security)](https://github.com/trailofbits/awesome-ml-security)
  * [AI's Predictable Path: 7 Things to Expect From AI in 2024+](https://danielmiessler.com/p/ai-predictable-path-7-components-2024)
  * [Artificial Intelligence and Cybersecurity](https://www.ismsforum.es/ficheros/descargas/isms-gt-ia-021707141605.pdf) (in Spanish :es:)
  * [Vigil](https://github.com/deadbits/vigil-llm): Detect prompt injections, jailbreaks, and other potentially risky Large Language Model (LLM) inputs. [![stars](https://badgen.net/github/stars/deadbits/vigil-llm)](https://github.com/deadbits/vigil-llm)
  * [Generative AI Models - Opportunities and Risks for Industry and Authorities](https://www.bsi.bund.de/SharedDocs/Downloads/EN/BSI/KI/Generative_AI_Models.pdf)
  * [Deploying AI Systems Securely. Best Practices for Deploying Secure and Resilient AI Systems](https://media.defense.gov/2024/Apr/15/2003439257/-1/-1/0/CSI-DEPLOYING-AI-SYSTEMS-SECURELY.PDF)
  * [NIST AI 600-1: Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile](https://airc.nist.gov/docs/NIST.AI.600-1.GenAI-Profile.ipd.pdf)
  * [:fr: ANSSI: Recommandations De Sécurité Pour Un Système d'IA Générative (Security Recommendations for a Generative AI System)](https://cyber.gouv.fr/sites/default/files/document/Recommandations_de_s%C3%A9curit%C3%A9_pour_un_syst%C3%A8me_d_IA_g%C3%A9n%C3%A9rative.pdf)
  * [PyRIT](https://github.com/Azure/PyRIT): The Python Risk Identification Tool for generative AI (PyRIT) is an open-access automation framework to empower security professionals and machine learning engineers to proactively find risks in their generative AI systems. [![stars](https://badgen.net/github/stars/Azure/PyRIT)](https://github.com/Azure/PyRIT)
  * [OWASP-Agentic-AI](https://github.com/precize/OWASP-Agentic-AI): Working to create the OWASP Top 10 for Agentic AI (AI Agent Security). [![stars](https://badgen.net/github/stars/precize/OWASP-Agentic-AI)](https://github.com/precize/OWASP-Agentic-AI)
  * [Towards Guaranteed Safe AI: A Framework for Ensuring Robust and Reliable AI Systems](https://arxiv.org/abs/2405.06624)
  * [Defining Real AI Risks](https://josephthacker.com/ai/2024/05/19/defining-real-ai-risks.html)
  * [Secure approach to generative AI](https://aws.amazon.com/es/ai/generative-ai/security/)
  * [Large Language Models in Cybersecurity](https://link.springer.com/content/pdf/10.1007/978-3-031-54827-7.pdf)
  * [Hey, That's My Model! Introducing Chain & Hash, An LLM Fingerprinting Technique](https://arxiv.org/abs/2407.10887)
  * [Generative AI Misuse: A Taxonomy of Tactics and Insights from Real-World Data](https://arxiv.org/abs/2406.13843)
  * [AI Risk Repository](https://airisk.mit.edu/)
  * [Revisiting AI Red-Teaming](https://cset.georgetown.edu/article/revisiting-ai-red-teaming/)
  * [German-French recommendations for the use of AI programming assistants](https://www.bsi.bund.de/SharedDocs/Downloads/EN/BSI/KI/ANSSI_BSI_AI_Coding_Assistants.html)
  * [Scalable watermarking for identifying large language model outputs](https://www.nature.com/articles/s41586-024-08025-4)

## 📊 Surveys 📊

  * [The Threat of Offensive AI to Organizations](https://arxiv.org/abs/2106.15764)
  * [Artificial Intelligence in the Cyber Domain: Offense and Defense](https://www.mdpi.com/2073-8994/12/3/410)
  * [A survey on adversarial attacks and defenses](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cit2.12028)
  * [Adversarial Deep Learning: A Survey on Adversarial Attacks and Defense Mechanisms on Image Classification](https://ieeexplore.ieee.org/document/9895425)
  * [A Survey of Privacy Attacks in Machine Learning](https://arxiv.org/abs/2007.07646)
  * [Towards Security Threats of Deep Learning Systems: A Survey](https://arxiv.org/abs/1911.12562)
  * [A Survey on Security Threats and Defensive Techniques of Machine Learning: A Data-Driven View](https://ieeexplore.ieee.org/document/8290925)
  * [SoK: Security and Privacy in Machine Learning](https://ieeexplore.ieee.org/document/8406613)
  * [Adversarial Machine Learning: The Rise in AI-Enabled Crime and its Role in Spam Filter Evasion](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4155496)
  * [Threats, Vulnerabilities, and Controls of Machine Learning Based Systems: A Survey and Taxonomy](https://arxiv.org/abs/2301.07474)
  * [Adversarial Attacks and Defences: A Survey](https://arxiv.org/pdf/1810.00069.pdf)
  * [Security Matters: A Survey on Adversarial Machine Learning](https://arxiv.org/abs/1810.07339)
  * [A Survey on Adversarial Attacks for Malware Analysis](https://arxiv.org/pdf/2111.08223.pdf)
  * [Adversarial Machine Learning in Image Classification: A Survey Towards the Defender’s Perspective](https://arxiv.org/pdf/2009.03728.pdf)
  * [A Survey of Robust Adversarial Training in Pattern Recognition: Fundamental, Theory, and Methodologies](https://arxiv.org/abs/2203.14046)
  * [Privacy in Large Language Models: Attacks, Defenses and Future Directions](https://arxiv.org/abs/2310.10383)

## 🗣 Maintainers  🗣

<table>
  <tr>
    <td align="center"><a href="https://github.com/Miguel000"><img src="https://avatars2.githubusercontent.com/u/13256426?s=460&v=4" width="150;" alt=""/><br /><sub><b>Miguel Hernández</b></sub></a></td>
    <td align="center"><a href="https://github.com/jiep"><img src="https://avatars2.githubusercontent.com/u/414463?s=460&v=4" width="150px;" alt=""/><br /><sub><b>José Ignacio Escribano</b></sub></a></td>
  </tr>
</table>

## ©️ License ©️

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

* [Creative Commons Attribution-Share Alike 4.0 International](LICENSE.txt)

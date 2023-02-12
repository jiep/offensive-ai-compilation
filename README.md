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
  * [deep-voice-conversion](https://github.com/andabi/deep-voice-conversion): Deep neural networks for voice conversion (voice style transfer) in Tensorflow. [![stars](https://badgen.net/github/stars/andabi/deep-voice-conversion)](https://github.com/andabi/deep-voice-conversion)
  * [tacotron](https://github.com/keithito/tacotron): A TensorFlow implementation of Google's Tacotron speech synthesis with pre-trained model (unofficial). [![stars](https://badgen.net/github/stars/keithito/tacotron)](https://github.com/keithito/tacotron)
  * [deepvoice3_pytorch](https://github.com/r9y9/deepvoice3_pytorch): PyTorch implementation of convolutional neural networks-based text-to-speech synthesis models. [![stars](https://badgen.net/github/stars/r9y9/deepvoice3_pytorch)](https://github.com/r9y9/deepvoice3_pytorch)
  * [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning): Clone a voice in 5 seconds to generate arbitrary speech in real-time. [![stars](https://badgen.net/github/stars/CorentinJ/Real-Time-Voice-Cloning)](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
  * [mimic2](https://github.com/MycroftAI/mimic2): Text to Speech engine based on the Tacotron architecture, initially implemented by Keith Ito. [![stars](https://badgen.net/github/stars/MycroftAI/mimic2)](https://github.com/MycroftAI/mimic2)
  * [Neural-Voice-Cloning-with-Few-Samples](https://github.com/Sharad24/Neural-Voice-Cloning-with-Few-Samples): Implementation of Neural Voice Cloning with Few Samples Research Paper by Baidu. [![stars](https://badgen.net/github/stars/Sharad24/Neural-Voice-Cloning-with-Few-Samples)](https://github.com/Sharad24/Neural-Voice-Cloning-with-Few-Samples)
  * [DiffSinger](https://github.com/MoonInTheRiver/DiffSinger): Singing Voice Synthesis via Shallow Diffusion Mechanism (SVS & TTS). [![stars](https://badgen.net/github/stars/MoonInTheRiver/DiffSinger)](https://github.com/MoonInTheRiver/DiffSinger)
  * [Vall-E](https://github.com/enhuiz/vall-e): An unofficial PyTorch implementation of the audio LM VALL-E. [![stars](https://badgen.net/github/stars/enhuiz/vall-e)](https://github.com/enhuiz/vall-e)
  * [TorToiSe](https://github.com/neonbjb/tortoise-tts): A multi-voice TTS system trained with an emphasis on quality. [![stars](https://badgen.net/github/stars/neonbjb/tortoise-tts)](https://github.com/neonbjb/tortoise-tts)
  * [🎸 Riffusion](https://github.com/riffusion/riffusion): Stable diffusion for real-time music generation. [![stars](https://badgen.net/github/stars/riffusion/riffusion)](https://github.com/riffusion/riffusion)
  * [whisper.cpp](https://github.com/ggerganov/whisper.cpp): Port of OpenAI's Whisper model in C/C++. [![stars](https://badgen.net/github/stars/ggerganov/whisper.cpp)](https://github.com/ggerganov/whisper.cpp)
  * 

#### Applications

#### Detection
  * [fake-voice-detection](https://github.com/dessa-oss/fake-voice-detection): Using temporal convolution to detect Audio Deepfakes. [![stars](https://badgen.net/github/stars/dessa-oss/fake-voice-detection)](https://github.com/dessa-oss/fake-voice-detection)


### Image

#### Tools

  * [StyleGAN](https://github.com/NVlabs/stylegan): StyleGAN - Official TensorFlow Implementation. [![stars](https://badgen.net/github/stars/NVlabs/stylegan)](https://github.com/NVlabs/stylegan)
  * [StyleGAN2](https://github.com/NVlabs/stylegan2): StyleGAN2 - Official TensorFlow Implementation. [![stars](https://badgen.net/github/stars/NVlabs/stylegan2)](https://github.com/NVlabs/stylegan2)
  * [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch): StyleGAN2-ADA - Official PyTorch implementation. [![stars](https://badgen.net/github/stars/NVlabs/stylegan2-ada-pytorch)](https://github.com/NVlabs/stylegan2-ada-pytorch) 
  * [StyleGAN3](https://github.com/NVlabs/stylegan3): Official PyTorch implementation of StyleGAN3. [![stars](https://badgen.net/github/stars/NVlabs/stylegan3)](https://github.com/NVlabs/stylegan3)
  * [Imaginaire](https://github.com/NVlabs/imaginaire): Imaginaire is a pytorch library that contains optimized implementation of several image and video synthesis methods developed at NVIDIA. [![stars](https://badgen.net/github/stars/NVlabs/imaginaire)](https://github.com/NVlabs/imaginaire)
  * [eg3d](https://github.com/NVlabs/eg3d): Efficient Geometry-aware 3D Generative Adversarial Networks. [![stars](https://badgen.net/github/stars/NVlabs/eg3d)](https://github.com/NVlabs/eg3d)
  * [ffhq-dataset](https://github.com/NVlabs/ffhq-dataset): Flickr-Faces-HQ Dataset (FFHQ). [![stars](https://badgen.net/github/stars/NVlabs/ffhq-dataset)](https://github.com/NVlabs/ffhq-dataset)
  * [DALLE2-pytorch](https://github.com/lucidrains/DALLE2-pytorch): Implementation of DALL-E 2, OpenAI's updated text-to-image synthesis neural network, in Pytorch. [![stars](https://badgen.net/github/stars/lucidrains/DALLE2-pytorch)](https://github.com/lucidrains/DALLE2-pytorch)
  * [ImaginAIry](https://github.com/brycedrennan/imaginAIry): AI imagined images. Pythonic generation of stable diffusion images. [![stars](https://badgen.net/github/stars/brycedrennan/imaginAIry)](https://github.com/brycedrennan/imaginAIry)
  * [instruct-pix2pix](https://github.com/timothybrooks/instruct-pix2pix): PyTorch implementation of InstructPix2Pix, an instruction-based image editing model. [![stars](https://badgen.net/github/stars/timothybrooks/instruct-pix2pix)](https://github.com/timothybrooks/instruct-pix2pix)
  * [Lama Cleaner](https://github.com/Sanster/lama-cleaner): Image inpainting tool powered by SOTA AI Model. Remove any unwanted object, defect, people from your pictures or erase and replace(powered by stable diffusion) any thing on your pictures. [![stars](https://badgen.net/github/stars/Sanster/lama-cleaner)](https://github.com/Sanster/lama-cleaner)
  * [Invertible-Image-Rescaling](https://github.com/pkuxmq/Invertible-Image-Rescaling): This is the PyTorch implementation of paper: Invertible Image Rescaling. [![stars](https://badgen.net/github/stars/pkuxmq/Invertible-Image-Rescaling)](https://github.com/pkuxmq/Invertible-Image-Rescaling)
  * [DifFace](https://github.com/zsyOAOA/DifFace): Blind Face Restoration with Diffused Error Contraction (PyTorch). [![stars](https://badgen.net/github/stars/zsyOAOA/DifFace)](https://github.com/zsyOAOA/DifFace)
  * [CodeFormer](https://github.com/sczhou/CodeFormer): Towards Robust Blind Face Restoration with Codebook Lookup Transformer. [![stars](https://badgen.net/github/stars/sczhou/CodeFormer)](https://github.com/sczhou/CodeFormer)
  * [Custom Diffusion](https://github.com/adobe-research/custom-diffusion): Multi-Concept Customization of Text-to-Image Diffusion. [![stars](https://badgen.net/github/stars/adobe-research/custom-diffusion)](https://github.com/adobe-research/custom-diffusion)
  * [Diffusers](https://github.com/huggingface/diffusers): 🤗 Diffusers: State-of-the-art diffusion models for image and audio generation in PyTorch. [![stars](https://badgen.net/github/stars/huggingface/diffusers)](https://github.com/huggingface/diffusers)
  * [Stable Diffusion](https://github.com/Stability-AI/stablediffusion): High-Resolution Image Synthesis with Latent Diffusion Models. [![stars](https://badgen.net/github/stars/Stability-AI/stablediffusion)](https://github.com/Stability-AI/stablediffusion)
  * [InvokeAI](https://github.com/invoke-ai/InvokeAI): InvokeAI is a leading creative engine for Stable Diffusion models, empowering professionals, artists, and enthusiasts to generate and create visual media using the latest AI-driven technologies. The solution offers an industry leading WebUI, supports terminal use through a CLI, and serves as the foundation for multiple commercial products. [![stars](https://badgen.net/github/stars/invoke-ai/InvokeAI)](https://github.com/invoke-ai/InvokeAI)
  * [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui): Stable Diffusion web UI. [![stars](https://badgen.net/github/stars/AUTOMATIC1111/stable-diffusion-webui)](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
  * [Stable Diffusion Infinity](https://github.com/lkwq007/stablediffusion-infinity): Outpainting with Stable Diffusion on an infinite canvas. [![stars](https://badgen.net/github/stars/lkwq007/stablediffusion-infinity)](https://github.com/lkwq007/stablediffusion-infinity)
  * [Fast Stable Diffusion](https://github.com/TheLastBen/fast-stable-diffusion): fast-stable-diffusion + DreamBooth. [![stars](https://badgen.net/github/stars/TheLastBen/fast-stable-diffusion)](https://github.com/TheLastBen/fast-stable-diffusion)
  * [GET3D](https://github.com/nv-tlabs/GET3D): A Generative Model of High Quality 3D Textured Shapes Learned from Images. [![stars](https://badgen.net/github/stars/nv-tlabs/GET3D)](https://github.com/nv-tlabs/GET3D)
  * [Awesome AI Art Image Synthesis](https://github.com/altryne/awesome-ai-art-image-synthesis): A list of awesome tools, ideas, prompt engineering tools, colabs, models, and helpers for the prompt designer playing with aiArt and image synthesis. Covers Dalle2, MidJourney, StableDiffusion, and open source tools. [![stars](https://badgen.net/github/stars/altryne/awesome-ai-art-image-synthesis)](https://github.com/altryne/awesome-ai-art-image-synthesis)
  * [Stable Diffusion](https://github.com/CompVis/stable-diffusion): A latent text-to-image diffusion model. [![stars](https://badgen.net/github/stars/CompVis/stable-diffusion)](https://github.com/CompVis/stable-diffusion)
  * [Weather Diffusion](https://github.com/IGITUGraz/WeatherDiffusion): Code for "Restoring Vision in Adverse Weather Conditions with Patch-Based Denoising Diffusion Models". [![stars](https://badgen.net/github/stars/IGITUGraz/WeatherDiffusion)](https://github.com/IGITUGraz/WeatherDiffusion)
  * [DF-GAN](https://github.com/tobran/DF-GAN): A Simple and Effective Baseline for Text-to-Image Synthesis. [![stars](https://badgen.net/github/stars/tobran/DF-GAN)](https://github.com/tobran/DF-GAN)
  * [Dall-E Playground](https://github.com/saharmor/dalle-playground): A playground to generate images from any text prompt using Stable Diffusion (past: using DALL-E Mini). [![stars](https://badgen.net/github/stars/saharmor/dalle-playground)](https://github.com/saharmor/dalle-playground)
  * [StyleNeRF](https://github.com/facebookresearch/StyleNeRF): This is the open source implementation of the ICLR2022 paper "StyleNeRF: A Style-based 3D-Aware Generator for High-resolution Image Synthesis". [![stars](https://badgen.net/github/stars/facebookresearch/StyleNeRF)](https://github.com/facebookresearch/StyleNeRF)


#### Applications

#### Detection

  * [stylegan3-detector](https://github.com/NVlabs/stylegan3-detector): StyleGAN3 Synthetic Image Detection. [![stars](https://badgen.net/github/stars/NVlabs/stylegan3-detector)](https://github.com/NVlabs/stylegan3-detector)

### Video

#### Tools
  * [DeepFaceLab](https://github.com/iperov/DeepFaceLab): DeepFaceLab is the leading software for creating deepfakes. [![stars](https://badgen.net/github/stars/iperov/DeepFaceLab)](https://github.com/iperov/DeepFaceLab)
  * [faceswap](https://github.com/deepfakes/faceswap): Deepfakes Software For All. [![stars](https://badgen.net/github/stars/deepfakes/faceswap)](https://github.com/deepfakes/faceswap)
  * [dot](https://github.com/sensity-ai/dot): The Deepfake Offensive Toolkit. [![stars](https://badgen.net/github/stars/sensity-ai/dot)](https://github.com/sensity-ai/dot)
  * [SimSwap](https://github.com/neuralchen/SimSwap): An arbitrary face-swapping framework on images and videos with one single trained model! [![stars](https://badgen.net/github/stars/neuralchen/SimSwap)](https://github.com/neuralchen/SimSwap)
  *  [faceswap-GAN](https://github.com/shaoanlu/faceswap-GAN): A denoising autoencoder + adversarial losses and attention mechanisms for face swapping. [![stars](https://badgen.net/github/stars/shaoanlu/faceswap-GAN)](https://github.com/shaoanlu/faceswap-GAN)
  *  [face2face-demo](https://github.com/datitran/face2face-demo): pix2pix demo that learns from facial landmarks and translates this into a face. [![stars](https://badgen.net/github/stars/datitran/face2face-demo)](https://github.com/datitran/face2face-demo)
  *  [Faceswap-Deepfake-Pytorch](https://github.com/Oldpan/Faceswap-Deepfake-Pytorch): Faceswap with Pytorch or DeepFake with Pytorch. [![stars](https://badgen.net/github/stars/Oldpan/Faceswap-Deepfake-Pytorch)](https://github.com/Oldpan/Faceswap-Deepfake-Pytorch)
  *  [Point-E](https://github.com/openai/point-e): Point cloud diffusion for 3D model synthesis. [![stars](https://badgen.net/github/stars/openai/point-e)](https://github.com/openai/point-e)

#### Applications

#### Detection
  * [FaceForensics++](https://github.com/ondyari/FaceForensics): FaceForensics dataset. [![stars](https://badgen.net/github/stars/ondyari/FaceForensics)](https://github.com/ondyari/FaceForensics)
  * [DeepFake-Detection](https://github.com/dessa-oss/DeepFake-Detection): Towards deepfake detection that actually works. [![stars](https://badgen.net/github/stars/dessa-oss/DeepFake-Detection)](https://github.com/dessa-oss/DeepFake-Detection)
  * [fakeVideoForensics](https://github.com/bbvanexttechnologies/fakeVideoForensics): Detect deep fakes videos. [![stars](https://badgen.net/github/stars/bbvanexttechnologies/fakeVideoForensics)](https://github.com/bbvanexttechnologies/fakeVideoForensics)
  * [Deepfake-Detection](https://github.com/HongguLiu/Deepfake-Detection): The Pytorch implemention of Deepfake Detection based on Faceforensics++. [![stars](https://badgen.net/github/stars/HongguLiu/Deepfake-Detection)](https://github.com/HongguLiu/Deepfake-Detection)
  * [SeqDeepFake](https://github.com/rshaojimmy/SeqDeepFake): PyTorch code for SeqDeepFake: Detecting and Recovering Sequential DeepFake Manipulation. [![stars](https://badgen.net/github/stars/rshaojimmy/SeqDeepFake)](https://github.com/rshaojimmy/SeqDeepFake)
  * [PCL-I2G](https://github.com/jtchen0528/PCL-I2G): Unofficial Implementation: Learning Self-Consistency for Deepfake Detection. [![stars](https://badgen.net/github/stars/jtchen0528/PCL-I2G)](https://github.com/jtchen0528/PCL-I2G)

### Text

#### Tools
  * [GLM-130B](https://github.com/THUDM/GLM-130B): An Open Bilingual Pre-Trained Model. [![stars](https://badgen.net/github/stars/THUDM/GLM-130B)](https://github.com/THUDM/GLM-130B)
  * [LongtermChatExternalSources](https://github.com/daveshap/LongtermChatExternalSources): GPT-3 chatbot with long-term memory and external sources. [![stars](https://badgen.net/github/stars/daveshap/LongtermChatExternalSourcess)](https://github.com/daveshap/LongtermChatExternalSourcess)
  * [sketch](https://github.com/approximatelabs/sketch): AI code-writing assistant that understands data content. [![stars](https://badgen.net/github/stars/approximatelabs/sketch)](https://github.com/approximatelabs/sketch)
  * [LangChain](https://github.com/hwchase17/langchain): ⚡ Building applications with LLMs through composability ⚡. [![stars](https://badgen.net/github/stars/hwchase17/langchain)](https://github.com/hwchase17/langchain)
  * [ChatGPT Wrapper](https://github.com/mmabrouk/chatgpt-wrapper): API for interacting with ChatGPT using Python and from Shell. [![stars](https://badgen.net/github/stars/mmabrouk/chatgpt-wrapper)](https://github.com/mmabrouk/chatgpt-wrapper) 
  * [openai-python](https://github.com/openai/openai-python): The OpenAI Python library provides convenient access to the OpenAI API from applications written in the Python language. [![stars](https://badgen.net/github/stars/openai/openai-python)](https://github.com/openai/openai-python)
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
  * 

#### Detection

#### Applications

### Misc

  * [🚀 Awesome Reinforcement Learning for Cyber Security](https://github.com/Limmen/awesome-rl-for-cybersecurity): A curated list of resources dedicated to reinforcement learning applied to cyber security. [![stars](https://badgen.net/github/stars/Limmen/awesome-rl-for-cybersecurity)](https://github.com/Limmen/awesome-rl-for-cybersecurity)
  * [Awesome Machine Learning for Cyber Security](https://github.com/jivoi/awesome-ml-for-cybersecurity): A curated list of amazingly awesome tools and resources related to the use of machine learning for cyber security. [![stars](https://badgen.net/github/stars/jivoi/awesome-ml-for-cybersecurity)](https://github.com/jivoi/awesome-ml-for-cybersecurity)
  * [Hugging Face Diffusion Models Course](https://github.com/huggingface/diffusion-models-class): Materials for the Hugging Face Diffusion Models Course. [![stars](https://badgen.net/github/stars/huggingface/diffusion-models-class)](https://github.com/huggingface/diffusion-models-class)


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
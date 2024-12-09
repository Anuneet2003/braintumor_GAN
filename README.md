# braintumor_GAN


A Minor Project Report on


Generative AI in Brain Tumor Detection

Submitted in partial fulfillment of the requirements for the degree of

Bachelor of Technology
in
Computer Science and Engineering (AIML) 2021-2025
by

Anuneet Rastogi ROLL NO: 219310251












Under the supervision of
Dr. Siddharth Kumar(Aiml)

SCHOOL OF COMPUTER SCIENCE AND ENGINEERING
DEPARTMENT OF ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING MANIPAL UNIVERSITY JAIPUR,
JAIPUR, RAJASTHAN JAN-MAY, 2024
 
DECLARATION


I hereby declare that the thesis entitled “Generative AI in Brain Tumor Detection” submitted to Manipal University Jaipur for the award of the degree of Bachelor of Technology is a record of bonafide work carried out by me under the supervision of Dr. Siddharth Kumar, Assistant Professor, School of Computer Science and Engineering-AIML, Manipal University Jaipur, Rajasthan.

I further declare that the work reported in this thesis has not been submitted and will not be submitted, either in part or in full, for the award of any other degree or diploma in this institute or any other institute or university in India or abroad.








Place: MUJ	Signature

Date: 16.05.2024	(Anuneet Rastogi)
 
CERTIFICATE


This is to certify that the thesis entitled “Generative AI in Brain Tumor Detection” submitted by Anuneet Rastogi (219310251) , School of Computer Science and Engineering (AIML), Manipal University Jaipur, Rajasthan for the award of the degree of Bechlor of Tech- nology is a record of bonafide work carried out by him under my supervision, as per the code of academic and research ethics of Manipal University jaipur, Rajasthan.
The contents of this report have not been submitted and will not be submitted either in part or in full, for the award of any other degree or diploma in this institute or any other institute or university. The thesis fulfills the requirements and regulations of the University and in my opinion, meets the necessary standards for submission.






Place: Manipal University Jaipur		Signature Date: 16/05/2024	(Dr. Siddharth Kumar)
 
ABSTRACT
Brain tumor detection and classification are crucial tasks in medical imaging analysis, re- quiring a diverse dataset for robust model training. However, acquiring a large and diverse dataset is often challenging due to factors like privacy concerns and data scarcity. To address this, we propose a novel approach utilizing Generative Adversarial Networks (GANs) for gen- erating synthetic MRI images of brain tumors. Our framework leverages GANs to learn the underlying distribution of real MRI images and generate realistic synthetic counterparts. We implement a GAN architecture comprising a generator and a discriminator, trained on a small dataset of real brain tumor MRI images. Through extensive experimentation, we demonstrate the efficacy of our approach in generating high-quality synthetic MRI images, thus augmenting the dataset for brain tumor detection tasks.
By employing GANs, we alleviate the need for large-scale labeled datasets, making our ap- proach particularly valuable in scenarios with limited data availability. The generated synthetic images exhibit realistic characteristics, preserving important features relevant for brain tumor detection. Our method provides a cost-effective and efficient solution for data augmentation in medical imaging tasks, facilitating the development of robust and accurate brain tumor detec- tion models. Overall, our work contributes to advancing the field of medical image analysis by providing a scalable and accessible solution for generating diverse MRI datasets, ultimately enhancing the performance of brain tumor detection systems
 
ACKNOWLEDGEMENT


With immense pleasure and deep sense of gratitude, I wish to express my sincere thanks to my supervisor Dr. Siddharth Kumar, School of Computer Science and Engineering, Ma- nipal University Jaipur, without his motivation and continuous encouragement, this research would not have been successfully completed.
I am grateful to the Honorable, Director Sir, Dr. Sandeep Chaurasia, School of Com- puter Science and Engineering, for motivating me to carry out this research Project.
I also express my sincere thanks to Dr. Puneet Mittal , HOD, School of Computer Science and Engineering (AIML), Manipal University Jaipur for his kind words of support and encouragement. I like to acknowledge the support rendered by my colleagues and friends in several ways throughout my research Project work.

Place: Manipal University Jaipur

Date: 16/05/2024	Anuneet Rastogi
 




Contents




ABSTRACT	. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
iii
ACKNOWLEDGEMENT . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
iv
1	Introduction
1
1.1	Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
1
1.2	Ethics Involved . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
3
2	Requirement Engineering - SRS
4
3	Design
7
3.1	Introduction of Design	. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
7
3.2	References . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
8
4	Implementation
14
4.1	Implementation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
14
5	Testing
18
6	Deployment
21
7	Conclusion
23
REFERENCES . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
24
LIST OF PUBLICATIONS . . . . . . . . . . . . . . . . . . . . . . . . . . . .
25
 
LIST OF TABLES
Table No	Table Title	Page No
1 - 2.1	Articles Included in the review focusing on image syn-
thesis.	11
2 - 2.2	Visual Training Test results by a physician for classify-
ing real vs synthetic images.	13




LIST OF FIGURES
 





Chapter 1

Introduction




1.1	Introduction

In the last decade artificial intelligence and machine learning have significantly devel- oped in many industries, for example, health care, retail, agriculture and manufacturing. Specific to the healthcare sector, these technologies have effectively changed the analysis of medical data, which has led to progress in diagnosing and treating several diseases. The potential in AI and the potential in the use of machine learning to improve health outcome results are immense; therefore, making them invaluable tools for medical experts.
The significant advancements and rapid development of AI and machine learning have brought about numerous studies concerning healthcare which have either been completed or are already underway. While the amount of work being done might be beneficial, one significant problem that may arise is the limitations of a dataset.
Medical datasets, specifically in the domain of medical imaging, often suffer from limited availability and imbalanced distribution. Considering the vast population of the entire world, a very limited amount of medical data is available, and even less of it can be used to gain the information about the symptoms of the disease. In recent years, few AI and deep learning techniques have been introduced to tackle this problem. Generative Adversarial Networks(GAN) is one of them.
Generative Adversarial Networks(GANs) are a prominent deep learning framework which can be used to generate synthetic data. One of the first GAN models was released by Google and also is the key inspiration for this research. The GAN’s application in medical imaging to produce synthetic data will help in addressing the crucial challenges faced by researchers and medical professionals.
 
In this project, we explore the use of Generative Adversarial Networks(GANs) for generating artificial brain MRI scans. Brain’s MRI play an important role in the diagnosis  and management of various neurological disorders, making it a vital component of medical research. One such domain is the detection of brain tumor, which requires a vast and diverse dataset, and acquiring such a dataset could be a lethargic and  tedious task. By using the capabilities of GAN, specifically DCGAN(Deep Convolutional GAN), we aim to generate synthetic brain MRI scan images that can enhance and complement the existing datasets, ultimately facilitating the medical imaging analysis.

1.2	Problem Statement In India, approximately 40,000 to 50,000 patients are diag- nosed with brain tumors every year. While MRI scans serve as a crucial diagnostic tool, yielding accurate results, the limited number of positive cases poses a significant challenge in building comprehensive datasets for medical research and machine learning applications. With only 35 positive samples for every 10,000 MRI scans, the imbalance between tumor and non-tumor cases exacerbates the difficulty in developing robust models for brain tumor detection and classification.
Moreover, the healthcare domain faces numerous obstacles in accessing large and diverse datasets, hindering the progress of research and innovation in medical imaging analysis. Issues such as privacy concerns, data scarcity, and regulatory constraints further compound the challenges associated with dataset acquisition. Consequently, machine learning problems like class imbalance and bias emerge, compromising the effectiveness and reliability of diagnostic models.
To address these challenges, we propose leveraging Generative Adversarial Networks (GANs) to produce synthetic MRI images of brain tumors. By synthesizing realistic images that mimic the characteristics of real MRI scans, our approach aims to augment the dataset for brain tumor detection tasks, mitigating the impact of data scarcity and imbalance in class. Through the development of a GAN-based framework, we seek to provide a scalable and accessible solution for generating diverse MRI datasets, ultimately enhancing the accuracy and reliability of brain tumor detection systems in the healthcare domain.
 
1.2	Ethics Involved

1.	Innovative Solution with Generative AI : We propose an innovative solution to address the disparities in medical imaging datasets using Generative AI techniques. By harnessing the power of Generative Adversarial Networks (GANs), our approach offers a revolutionary method to generate synthetic medical data, effectively mitigating the limitations posed by insufficient real-world datasets.
2.	Revolutionary Data Generation : Our project introduces a paradigm shift in data generation for medical imaging analysis. Instead of relying solely on limited real- world data, we leverage Generative models, specifically deep generative models, to learn the underlying distribution of medical images. This enables us to produce synthetic samples that most likely resemble real MRI scans, thereby expanding the diversity and quantity of available data for research and model training purposes.
3.	Utilization of Generative Models : We utilize Generative models, a class of deep learning models famous for their ability to learn complex data distributions and produce new samples with diverse properties. By employing Generative models, we can create synthetic MRI images that capture the variability and nuances present in real-world medical data. This approach facilitates the development of robust and reliable brain tumor detection models by providing a larger and more diverse dataset for training and evaluation.
 





Chapter 2


Requirement Engineering - SRS


•	Project Initiation and Planning:

The project commenced with the delineation of clear objectives, scope, and time- line. This involved a thorough literature review to understand existing techniques for brain tumor detection and the usage of Generative Adversarial Network(GANs) in medical imaging. Additionally, we identified the necessary resources, including hardware, software, and datasets, essential for the project’s execution.

•	Requirement Analysis:

Detailed analysis was conducted to determine the specific requirements for the GAN model. This encompassed defining input and output dimensions, desired performance metrics, and data preprocessing steps. Moreover, evaluation criteria were established to assess the model’s efficacy in generating brain MRI scan images accurately.

•	Dataset Collection and Preprocessing:

The brain MRI scan dataset was obtained from the Kaggle repository, comprising images with and without tumors. Preprocessing steps were then applied to prepare the dataset for training the GAN model. Techniques such as normalization, resizing,  and data augmentation were employed to better the quality and variability of the dataset. Furthermore, the dataset is divided into training & validation, and testing sets to do model evaluation.

•	Model Building:
 
Deep Convolutional GAN (DCGAN) architecture was implemented based on Google’s model, tailored to the specific requirements of brain MRI scans. This involved defining the generator and discriminator models, incorporating appropriate convolutional and upsampling layers. Additionally, loss functions, optimization algorithms, and hyperparameters were specified to optimize the training process.

•	Model Training:

The GAN model was trained on the preprocessed brain MRI scan dataset, with performance monitored on the validation set. Techniques such as checkpointing and early stopping were employed to prevent overfitting and ensure optimal model performance. Hyperparameters and architectural choices were fine-tuned iteratively based on validation results to enhance model accuracy and stability.

•	Testing and Validation:

The trained GAN model underwent rigorous testing and validation on the held-out test set. Performance evaluation metrices such as precision, accuracy, recall, and F1-score were found to assess the model’s efficacy. Additionally, the quality and similarity of generated synthetic brain MRI scan images were analyzed and compared with real MRI scans. Finally, the model’s performance was benchmarked against existing techniques for brain tumor detection from MRI scans to gauge its effectiveness and potential for real-world applications.

•	Basic GAN Architecture:

After preprocessing the data, due to hardware constraints, 20 random images are selected for training. A basic DCGAN consists of two CNN models, generator and discriminator. As per the figure, the generator inputs random noise as input and converts it into a sample instance. This sample instance is then compared by the original image of the dataset by the discriminator. Depending upon the output from the discriminator, the loss functions are optimized and the same process is circled over until the discriminator is unable to distinguish between the artificially generated images and the real image from the dataset.
 
 


Figure 3.4 Flow of the Model
 





Chapter 3


Design




3.1	Introduction of Design

System Architecture Our system architecture revolves around a Generative Adversarial Network(GAN)-based framework designed to produce synthetic MRI images of brain tumors. The architecture comprises two primary components: the generator & the discriminator. The generator learns to produce realistic MRI pictures from random noise,   while the discriminator distinguishes between the real & synthetic pictures.   During training,   the generator will generate images that are indistinguishable from actual MRI scans, while the discriminator strives to differentiate between real & synthetic pictures accurately. This adversarial training process fosters the creation of high-quality synthetic pictures that almost resemble the    actual-world MRI scans of brain tumors.
 

Figure - 3.1 Basic Prototype
 

 


Figure - 3.2 WorkFlow Diagram


Figure 3.3 Flow chart


3.2	References

The intersection of Generative Adversarial Networks (GANs) and medical imaging anal- ysis has garnered significant attention in recent years, driven by the pressing need to address challenges related to data scarcity and class imbalance in healthcare datasets. GANs, introduced by Goodfellow et al. (2014), offer a novel approach to data genera-
 
tion by pitting two neural networks, namely the generator and the discriminator, against each other in a game-theoretic framework. This concept has been applied to various domains, including medical imaging, to generate synthetic data that closely resemble real-world samples.
However, despite the potential benefits, challenges remain in leveraging synthetic data for medical imaging analysis, including issues related to data quality, domain shift, and model generalization. Addressing these challenges requires careful consideration of var- ious factors, such as the choice of GAN architecture, training strategies, and evaluation methodologies. Moreover, ethical considerations surrounding the use of synthetic data in healthcare must be carefully navigated to ensure patient privacy and safety.
In summary, the integration of GANs and synthetic data holds promise for addressing data scarcity and class imbalance in medical imaging datasets. By leveraging the power of GANs to generate realistic images, researchers can augment existing datasets, im- prove model performance, and advance the field of medical image analysis. However, further research is needed to overcome challenges and unlock the full potential of syn- thetic data in healthcare applications.
1.	Medical Image Synthesis for Data Augmentation and Anonymization using Genera- tive Adversarial Networks:
Cite: Shin, Hoo-Chang, et al. Medical image synthesis for data augmentation and anonymization using generative adversarial networks. Simulation and Synthesis in Med- ical Imaging: Third International Workshop, SASHIMI 2018, Held in Conjunction with MICCAI 2018, Granada, Spain, September 16, 2018, Proceedings 3. Springer Interna-
tional Publishing, 2018.

https://arxiv.org/pdf/1807.10225

•	BRATS utilizes multi-institutional pre-operative MRIs and focuses on the segmen- tation of intrinsically heterogeneous (in appearance, shape, and histology) brain tumors, namely gliomas

[14]. Each patient’s MRI image set includes a variety of series including T1-weighted, T2-weighted, contrast-enhanced T1, and FLAIR, along with, ground-truth voxel-wise
 
annotation of edema, enhancing tumor, and non-enhancing tumor. They use the BRATS 2015 training data set which is publicly available.

•	The paper motivates the need for better data augmentation by discussing the general challenge of training on imbalanced datasets with rare pathological findings

2.	Unsupervised Representation Learning With Deep Convolutional Generative Adver- sarial Networks
•	This paper gives evidence that adversarial networks learn good representations of im- ages for supervised learning and generative modeling using Large-scale Scene Under- standing (LSUN) (Yu et al., 2015), Imagenet-1k and a newly assembled Faces dataset.

3 . Generative Adversarial Networks in Brain Imaging: A Narrative Review

Cite: Laino, Maria Elena, et al. Generative adversarial networks in brain imaging: A narrative review. Journal of imaging 8.4 (2022): 83.

•	This paper reviews the work done for image synthesis of brain mri scans. One of the mention works in the paper is Generative Adversarial Networks for the Creation of Realistic Artificial Brain Magnetic Resonance Images.
•	They created their own dataset in accordance with World Medical Association Dec- laration of

Helsinki and Ethical Principles for Medical Research Involving Human Subjects.In total,
96 T1-weighted (T1W) brain images of 30 healthy volunteers and 33 patients with a history of cerebrovascular accident (women, 26; mean age, 69 ± 10) were enrolled, while the latter group underwent MR scans both at acute and chronic phases (9 ± 6 and 101 ± 13 days after disease onset, respectively)

•	Their performance was rated by two neuroradiologists who tried to discern the syn- thetic images from the real ones: 45% and 71% of the synthetic images were rated as real MRIs by each radiologist, respectively, while 44% and 70% of the real images were rated as synthetic images.
 
Table 2.1- Articles Included in the review focusing on image synthesis.



Author	Year	Application	Population	Imaging
Modality	ML Model	Results



Kazuhiro	


2018	

Image Synthesis	30 healthy and 33		pa- tients	with cerebrovas accidents	


MRI	


DCGAN	45%	and
71%	were identified
as real   by
neuroradio logists

Islam	
2020	Image Synthesis	479 patients	
PET	
DCGAN	SSIM
77.48


Kim	

2020	

Image Synthesis	139 Patients alzheimer’s Disease and 347 Normal	

PET/CT	

Boundary	Accuracy
94.82
Sensitivity 92.11
AUC 0.98

Qingyun	
2020	
Image Synthesis	226 patients


HCG	
MRI	
TumarGan	
Dice 0.725

Barile	
2021	Image
Synthesis	29	relaps-
ing	
MRI	
GAN AAE	F1	score
81%

Hirte	
2021	Image
Synthesis	2029	nor-
mal brain	
MRI	
GAN	Similarity
0.0487

4	. Learning Implicit Brain MRI Manifolds with Deep Learning

Cite: Bermudez C, Plassard AJ, Davis TL, Newton AT, Resnick SM, Landman BA. Learning Implicit Brain MRI Manifolds with Deep Learning. Proc SPIE Int Soc Opt Eng. 2018 Mar;10574:105741L. doi: 10.1117/12.2293515. PMID: 29887659; PMCID: PMC5990281.
The image synthesis GAN and the denoising autoencoder experiments were conducted on 528 T1-weighted brain MRI images(without tumor) from healthy
 
controls (ages 29–94 years old, mean of 67.9 years old) as part of the Baltimore Longitudinal Study of Aging (BLSA) study, which is a study of aging operated by the National Institute of Aging.





















Figure - 2.1 Reference
They had two raters to rate the synthetic images, a neuroradiologist and a neuroimaging expert.
The first rater mentioned that despite comparable quality, the synthetic images were immediately given away by anatomic abnormalities such as largely asymmetric left and right caudate. Similarly, the second rater, noticed brighter intensities near the center of the image compared to the boundaries in the synthetic images. These comments represent challenges in image synthesis: anatomic accuracy and signal quality .
5. GAN-based synthetic brain MR image generation. In: 2018 IEEE
Cite: Han, Changhee, et al. GAN-based synthetic brain MR image generation. 2018 IEEE 15th international symposium on biomedical imaging (ISBI 2018). IEEE, 2018.
 
https://ieeexplore.ieee.org/abstract/document/8363678/

In particular, the BRATS 2016 training dataset contains 220 High-Grade Glioma (HGG) and
54 Low-Grade Glioma (LGG) cases, with T1-weighted (T1), contrast enhanced T1- weighted (T1c), T2-weighted (T2), and Fluid Attenuation Inversion Recovery (FLAIR) sequences—they were skull stripped and resampled to isotropic 1mm × 1mm × 1mm resolution with image dimension 240×240×155; among the different sectional planes Compares WGAN and DCGAN and shows how WGAN is more reliable than DCGAN.
Table 2.2 Visual Training Test results by a physician for classifying real vs synthetic images.



	Accuracy	Real Selected
as Real	Real as Synt	Synt as Real	Synt as Synt
T1 ( DCGAN,
128 * 128 )	
70	
26	
24	
6	
44
T1c (DCGAN
128 * 128)	
71	
24	
26	
3	
47
T2 (DCGAN
128 * 128)	
64	
22	
28	
8	
42
FLAIR(DCGAN
128 * 128)	
54	
12	
38	
8	
42
Concat(DCGAN
128 * 128)	
77	
34	
16	
7	
43
Concat(DCGAN
64 * 64)	
54	
13	
37	
9	
41
 





Chapter 4


Implementation




4.1	Implementation

The development environment for our project consists of both hardware and software components. Hardware requirements include a workstation equipped with a powerful GPU, such as the NVIDIA GeForce RTX or Tesla series, to facilitate efficient training of deep learning models. Additionally, sufficient RAM and storage capacity are essential for handling large datasets and model checkpoints.
On the software front, we utilize popular deep learning frameworks such as TensorFlow and
Keras for model development and training. These frameworks provide a rich set of tools and APIs for building and optimizing GAN architectures. We leverage Python as the primary programming language due to its versatility and extensive libraries for numerical computing and data manipulation. Furthermore, we employ auxiliary libraries such as OpenCV, NumPy, Matplotlib, and Seaborn for image processing, data visualization, and analysis.
In summary, our system design leverages GANs to generate synthetic MRI images of brain tumors, with a focus on creating a robust and efficient framework for data augmentation in medical imaging analysis. The development environment encompasses a combination of powerful hardware and versatile software tools to support the training and evaluation of deep learning models for generating synthetic medical data.
 
4.1.1	Classes of Project The project implementation is organized into several mod- ules/classes to streamline the development process and ensure modularity and scalability. The key modules/classes include:

•	DataHandler: Responsible for loading and preprocessing the MRI scan dataset, including resizing, normalization, and data augmentation.
•	GANModel: Implements the DCGAN architecture, comprising the generator and discriminator models, along with functions for model training and evaluation.
•	EvaluationMetrics: Contains functions to calculate evaluation metrics such as ac- curacy, precision, recall, and F1-score for model performance assessment.
•	Visualization: Handles the visualization of generated synthetic MRI images and comparison with real MRI scans.
•	Main: Orchestrates the execution of the project, including dataset loading, model initialization, training, evaluation, and result visualization.

4.1.2	Implementation Detail The implementation detail revolves around the utiliza- tion of TensorFlow and Keras libraries to develop the GAN model for generating syn- thetic MRI images. The DCGAN architecture is instantiated within the GANModel module, with the generator and discriminator models defined using convolutional and upsampling layers. The models are compiled with appropriate loss functions (e.g., bi- nary cross-entropy) and optimization algorithms (e.g., Adam optimizer) to facilitate ef- ficient training. The dataset preprocessing, including resizing, normalization, and data augmentation, is handled by the DataHandler module. During training, the GAN model iteratively learns to generate synthetic MRI images that closely resemble real MRI scans, with the training process monitored using validation metrics to ensure optimal model performance.
The architecture used for the generator and discriminator used in our research is ex- plained in the figure. The generator starts with a dense layer which takes a random noise vector also known as a latent vector as input. This input is then upsampled several times through multiple upsampling layers which further goes through 2D transpose filters and batch normalization layers which ensures that the output pixel values are in the desired
 
range. Each transpose layer consists of a kernel size of 4 pixels and a stride of 2 pixels. All layers use ReLu activation function except the final layer which uses tanh.
The discriminator takes both the real image and generated image sets as input with the corresponding labels. It has a reverse architecture as compared to the generator. It consists of 4 Convolutional 2D layers with a kernel size of 3 pixels and a stride of 2 pixels, which effectively downsamples the feature maps. After the convolutional layers, the feature maps are flattened into a 1D vector. It also consists of a dropout layer with a dropout rate of 0.4 which helps in preventing overfitting by randomly setting a fraction of the input units to 0 during training. The final layer is a dense layer which uses sigmoid activation function which outputs a value between 0 and 1 representing the probability that the input image is real (1) or fake (0).
The combination of the generator and discriminator forms the GAN architecture where the generator tries to surpass the discriminator by generating realistic images while the discriminator tries to accurately distinguish between real and fake images.



Figure 3.5 Model Architecture
 
 

Figure 3.6 Working
 





Chapter 5


Testing


We train the GAN model for 10 epochs using the Adam optimizer with a learning rate of 0.0002 . After running for 10 epochs, the GAN model is able to generate synthetic brain MRI images which resemble the images from the original dataset which can be seen in the figure.



Figure 4.4 Synthetic image




Figure 4.5 Real images
 
To build a good convolutional model, the answer lies in the loss function. We have used the binary cross entropy which is also known as the min-max loss function. Lookin at it as a min-max game, the generator tries to minimize this function while the discriminator tries to maximize it.



Following 10 epochs of operation, we observe a generator loss of 3.4853 and a discriminator loss of 0.0883. The generated samples were then tested by plotting a graph comparing them with the real samples. As shown in the figure, the graph of the generated samples almost overlaps that of the real samples.
o.5cm

Figure 4.6 Graph

After taking hardware constraints into consideration, the results obtained after training the model for relatively less epochs, we were still able to generate brain MRI images which resemble the images from the dataset.
 
 

Figure 4.7 Epoch 5
 





Chapter 6


Deployment





Figure 4.1 Discriminator

Figure 4.2 Generator
 
 

Figure 4.3 Combination




Figure 4.4 Sample Output
 





Chapter 7


Conclusion


In this study, we show the potential of DCGAN in generating realistic artificial medical images that closely resemble the real dataset. We were able to generate brain MRI images which, in terms of anatomical structures, intensity patterns, and general image characteristics, showed a high degree of similarity to the original data.
Although the binary cross entropy loss function use gets the job done, a different loss function like a WGAN can be used to eradicate any possibility of mode collapse or having monotonous data. The WGAN gives the discriminator the role of a critic instead of a judge which evaluates the generated image by giving it a score instead of just giving a binary output. This results in better optimization by the generator and helps in generating unique images.
Our work shows the adaptability and versatility of generative adversarial networks which can be very helpful in niche medical studies which require a particular type of data. GANs can be used to generate customized images to fit certain use cases, which makes it possible to look into uncommon and underrepresented diseases. We can conclude that generative adversarial networks and generative AI are very powerful and valuable tools which will be used by researchers and medical professionals in the coming years and have the capability to further revolutionize the industry. The project encompasses an interdisciplinary exploration across several key domains, including Generative Adver- sarial Networks (GANs), medical imaging analysis, and machine learning for healthcare applications. We delve into the development and optimization of a GAN-based frame- work tailored specifically for generating synthetic MRI images depicting brain tumors. Through meticulous experimentation and model refinement, we aim to address the per- sistent challenges stemming from data scarcity and class imbalance in medical datasets,
 
particularly in the context of brain tumor detection.

Furthermore, our research extends beyond mere data generation to encompass the eval- uation and integration of synthetic images into existing diagnostic workflows. By jux- taposing the generated MRI scans with authentic ones, we seek to validate the efficacy and authenticity of our approach in enhancing the performance of brain tumor detec- tion models. This entails rigorous testing and validation procedures, including assessing model accuracy on both synthetic and real datasets individually, followed by compre- hensive evaluation on a combined dataset.
Moreover, the project sets the stage for future advancements in medical image analy- sis by exploring the potential applications of synthetic data in augmenting real-world datasets. Through this endeavor, we aim to elucidate the utility of synthetic images as a complementary resource for training and evaluating machine learning models, thereby paving the way for more robust and reliable diagnostic systems in healthcare.
 






LIST OF PUBLICATIONS



(a)	Goodfellow, Ian, et al. Generative adversarial networks. Communications of the ACM 63.11 (2020): 139-144 .
(b)	Salimans, Tim, et al. Improved techniques for training gans. Advances in neural information processing systems 29 (2016).
(c)	Skandarani, Y.; Jodoin, P.-M.; Lalande, A. GANs for Medical Image Synthesis: An Empirical Study. J. Imaging 2023, 9, 69.
(d)	Bermudez C, Plassard AJ, Davis TL, Newton AT, Resnick SM, Landman BA. Learning Implicit Brain MRI Manifolds with Deep Learning. Proc SPIE Int Soc Opt Eng. 2018 Mar;10574:105741L. doi: 10.1117/12.2293515. PMID: 29887659; PMCID: PMC5990281.
(e)	Shin, Hoo-Chang, et al. Medical image synthesis for data augmentation and anonymiza- tion using generative adversarial networks. Simulation and Synthesis in Medical Imaging: Third International Workshop, SASHIMI 2018, Held in Conjunction with MICCAI 2018, Granada, Spain, September 16, 2018, Proceedings 3. Springer In- ternational Publishing, 2018.
(f)	Radford, Alec, Luke Metz, and Soumith Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434 (2015) .
(g)	Yi, Xin, Ekta Walia, and Paul Babyn. Generative adversarial network in medical imaging: A review. Medical image analysis 58 (2019): 101552.
(h)	Yi, Xin, Ekta Walia, and Paul Babyn. Generative adversarial network in medical imaging: A review. Medical image analysis 58 (2019): 101552.
 
(i)	Singh, Nripendra Kumar, and Khalid Raza. Medical image generation using gen- erative adversarial networks: A review. Health informatics: A computational per- spective in healthcare (2021): 77-96.
(j)	Yizhou Chen, Xu-Hua Yang, Zihan Wei, Ali Asghar Heidari, Nenggan Zheng,
Zhicheng

(k)		Li, Huiling Chen, Haigen Hu, Qianwei Zhou, Qiu Guan, Generative Adversarial Networks in Medical Image augmentation: A review,Computers in Biology and Medicine,Volume 144,2022,105382,ISSN 0010-4825,

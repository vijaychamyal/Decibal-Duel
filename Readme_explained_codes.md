#TASK 1: Untitled6.ipynb
##Below is the explanation of the code of task 1. I have explained all the terms used and logic behind using these things.
##Its score on Kaggle is 0.9891
I have submitted this on 13th but it is showing late submission so not updated in leaderboard
Here is the summary and then I have explained in detailed about this below:
1.	Setup & Config: The code first installs libraries, mounts my Google Drive, and sets up all key parameters. This includes the audio sample rate, image size, model name (EfficientNetV2-M), and training settings (epochs, learning rate, etc.).
2.	Audio Pre-processing: It defines a function extract_mel that takes any .wav file, resamples it to a standard 32kHz, pads or trims it to exactly 5 seconds, and converts it into a Mel spectrogram (a 2D image representation of the audio).
3.	Caching for Speed: To avoid re-processing the audio files every time, it runs a cache_mels function. This saves all the generated spectrograms as fast-loading .npy files in a CACHE_DIR.
4.	Data Loading & Augmentation: A custom PyTorch Dataset is created. When loading data for training, it:
a.	Loads the pre-cached .npy spectrogram.
b.	Applies a strong chain of data augmentations (like spec_aug, mixup, and cutmix) to create new, varied training examples. This prevents the model from overfitting.
c.	Resizes the spectrogram "image" to 384x384 to fit the model's input.
5.	Balanced Training: The script uses a WeightedRandomSampler to handle class imbalance. This means it intentionally oversamples audio from rare classes and undersamples from common ones so the model gets a more balanced view.
6.	Model Training: It loads the pre-trained EfficientNetV2-M model and fine-tunes it on the spectrograms. It uses modern training techniques like:
	Mixed Precision (scaler): To speed up training and use less GPU memory.
	OneCycleLR Scheduler: To intelligently manage the learning rate for faster convergence.
	Label Smoothing: A regularization technique to make the model less overconfident.
7.	It validates the model after each epoch and saves only the version with the highest validation accuracy.
8.	Inference with TTA: For prediction on the test set, it loads the best-saved model. Instead of predicting just once, it uses Test-Time Augmentation (TTA). It creates 12 slightly different versions of each test image, gets 12 separate predictions, and then averages them for a final, more robust answer.
a.	Submission: Finally, it formats these predictions into a submission.csv file and provides a command to download it.



Now this is detailed explanation of the code :
 1.	Setup and imports :
•	It installs necessary third-party libraries: “timm” (pytorch image models), “torchaudio” (for audio processing), and “librosa” (audio analysis tool).
•	It mounts  Google Drive, allowing  script to access files stored there (dataset).
•	It imports all the standard libraries needed for the script: os for file paths, numpy for numerical operations, pandas for handling data, torch for the deep learning framework, tqdm for progress bars, and sklearn for data splitting.
•	Finally, it suppresses warnings to keep the output clean.

2.	File and Cache Configuration
•	ROOT points to the main project folder on the google drive.
•	TRAIN_DIR and TEST_DIR point to the subfolders containing the raw .wav files.
•	CACHE_DIR defines a folder on the local Colab machine . used for optimization: it will be used to store the pre-processed spectrograms so the script doesn't have to re-calculate them every time it runs, which is much faster.

3.	Class Definitions
•	CLASSES is  list of the five sound class names. C2I (Class-to-Index) is a dictionary that maps each class name (e.g., 'dog_bark') to a number (e.g., 0). This is necessary because models work with numbers, not text.
•	I2C (Index-to-Class) is the reverse dictionary, mapping the number back to the name (e.g., 0 to 'dog_bark'), which is useful for creating the final submission.
4.	Audio Parameters
•	SR: The Sample Rate (32,000 Hz). All audio will be resampled to this rate for consistency.
•	DURATION: All audio clips will be standardized to 5.0 seconds.
•	N_MELS, N_FFT, HOP: These are technical parameters for creating the Mel spectrograms, defining the image's "height" (256 mel bands), the window size for the Fourier transform (2048), and the step size (320).
•	SAMPLES: Calculates the total number of audio samples that a 5-second clip at 32kHz should have (160,000)

5.	Training Hyperparameters
•	BATCH_SIZE: The model will look at 24 samples at a time.
•	IMG_SIZE: The spectrograms will be resized to 384x384 "images".
•	MODEL_NAME: Specifies the exact pre-trained model to use from the timm library: EfficientNetV2-M.
•	EPOCHS & LR: The training will run for 22 full cycles (epochs) with a starting Learning Rate of 0.00018. I have done so to increase accuracy
•	MIXUP_ALPHA & CUTMIX_ALPHA: Parameters for two data augmentation techniques (Mixup, Cutmix) that help prevent overfitting.
•	LABEL_SMOOTH: A regularization technique that makes the model less overconfident.
•	TTA_COUNT: For Test-Time Augmentation, meaning it will create 12 modified versions of each test sample and average the predictions for a more robust answer.
•	DEVICE: Automatically selects the GPU ('cuda') if it's available.
•	SEED: A fixed number (42) used to initialize all random number generators, ensuring the results are reproducible.
6.	Reproducibility Function: The seed_everything function sets the random seed for all relevant libraries (random, numpy, and torch). This is immediately called to ensure that any operation involving randomness (like data shuffling or model initialization) will be the same every time the script is run.

7.	Mel Spectrogram Extraction: It takes a file path and:
•	Loads the audio file, ensures it's mono (not stereo).
•	Resamples it to the target SR (32kHz).
•	Pads or truncates it to the exact 5-second SAMPLES length.
•	Computes the mel spectrogram using the parameters above.
•	Converts the power spectrogram to the decibel (dB) scale.
•	Normalizes the spectrogram (subtracts mean, divides by standard deviation) so all values are in a consistent range.
•	Clamps the values to prevent extreme outliers and returns it as a NumPy array.

8.	Caching Function: This function cache_mels iterates through a list of audio files. For each file, it checks if a pre-processed .npy version already exists in the CACHE_DIR.
•	If it exists: It just notes the path.
•	If it doesn't exist: It calls the extract_mel function to create the spectrogram and then saves the resulting NumPy array to the cache. This is for one-time running only. 

9.	Loading Training File Paths and Labels: This block scans the TRAIN_DIR. It loops through each class folder (e.g., 'dog_bark'), finds all .wav files inside it, and adds the file paths to train_files and their corresponding numeric labels (e.g., 0) to train_labels.

10.	Data Splitting: 
•	A training set (tr_files, tr_labels) with 92.5% of the data.
•	A validation set (vl_files, vl_labels) with 7.5% of the data. The stratify=train_labels argument is important: it ensures that the 7.5% split contains the same percentage of each class as the full dataset. It also gathers all the test file paths.
11.	Executing the Caching: This block actually runs the cache_mels function on the three file lists (train, validation, and test). When you run this, you will see progress bars as it processes and saves all the spectrograms to the CACHE_DIR.
12.	Spectrogram Augmentation:
•	spec_aug: Implements SpecAugment, which randomly masks out (sets to the mean) horizontal "frequency" bands and vertical "time" bands. This forces the model to learn more robust features.
•	aug_pipeline: This is a wrapper that applies a chain of random augmentations to a training spectrogram: it might apply spec_aug, add random noise, randomly scale the intensity, or shuffle time segments.

13.	PyTorch Dataset Class:
•	__init__: Stores the paths to the cached .npy files and the labels.
•	__getitem__: This method is called for every single sample. It:
•	Loads the .npy spectrogram from the cache.
•	If it's a training sample (self.augment is True), it applies the aug_pipeline.
•	It stacks the 1-channel spectrogram into 3 channels (to mimic an RGB image, which the pre-trained model expects).
•	It resizes the image to the target IMG_SIZE (384x384) using "bicubic" interpolation.
•	It returns the final image tensor and its label.

14.	Mixup and Cutmix Functions: These functions implement more advanced augmentations that work on an entire batch of images.
•	mixup: Takes two images and "blends" them together (e.g., 70% of image A + 30% of image B). The label is also blended (70% label A + 30% label B).
•	cutmix: Takes a patch from one image and "pastes" it onto another. The label is a mix based on the area of the patch. Both techniques are very effective at preventing overfitting and improving model generalization.
15.	Weighted Sampling:
•	It counts the number of samples in each class.
•	It creates a "weight" for every single training sample. Samples from rare classes get a high weight, and samples from common classes get a low weight.
•	It creates a WeightedRandomSampler that will build batches by "sampling" from the dataset according to these weights. This ensures the model sees a more balanced mix of classes during training.
16.	DataLoaders : Here, the AudioDataset classes are instantiated for train, validation, and test. Then, they are wrapped in DataLoaders. The DataLoader is what efficiently manages loading the data in batches, using the sampler for the training set, and using multiple CPU cores (num_workers) to prepare data in the background.
17.	Model, Optimizer, and Loss:
•	  model: Creates the EfficientNetV2-M model, loads the pre-trained weights, and replaces the final layer with a new one that outputs 5 classes.
•	 optimizer: AdamW is the algorithm that will update the model's weights to minimize the loss.
•	scheduler: OneCycleLR is a modern learning rate scheduler that will intelligently increase and then decrease the LR during training, which often leads to faster convergence.
•	 criterion: The loss function (CrossEntropyLoss), which measures how "wrong" the model's predictions are. It includes the LABEL_SMOOTH parameter.
•	 scaler: A GradScaler is initialized. This is used for mixed-precision training, a technique that uses less memory and speeds up training on modern GPUs.
18.	Training Loop Initialization:  It sets a variable best_acc to 0.0 to keep track of the best validation accuracy achieved. It starts the main for loop that will run for EPOCHS (22) times. Inside the loop, it puts the model in model.train() mode (which enables features like dropout) and resets the loss and accuracy counters for the new epoch.
19.	Training Batch Loop: This is the inner loop that processes each batch of training data. For each batch:
•	It moves the images (xb) and labels (yb) to the GPU.
•	It randomly applies Mixup or Cutmix (or neither).
•	It clears any old gradients (optimizer.zero_grad()).
•	with autocast(): This enables mixed-precision.
•	pred = model(xb): The forward pass, where the model makes predictions.
•	loss = ...: The loss is calculated. This line cleverly handles the blended labels from Mixup/Cutmix.
•	The scaler is used to perform the backward pass (loss.backward()) and update the model's weights (optimizer.step()) in a mixed-precision-safe way.
•	The learning rate scheduler is updated at every step.
•	The loss and accuracy for the batch are recorded.
20.	Validation Loop:
•	The model is put in model.eval() mode (which disables dropout and batch normalization updates).
•	with torch.no_grad(): This disables gradient calculation, which saves memory and speeds up the process since gradients aren't needed for validation.
•	It loops through all batches in the val_dl, makes predictions, and counts the number of correct ones. No augmentations or optimizers are used.
21.	Epoch End and Model Saving: At the end of each epoch:
•	The final validation accuracy (val_acc) is calculated.
•	A summary line is printed showing the training and validation accuracy.
•	It checks if the current val_acc is better than the best_acc seen so far.
•	If it is, it updates best_acc and saves the model's weights (model.state_dict()) to the file best_model_opt.pth. This ensures that you always keep the single best version of the model.
22.	Inference (TTA) Setup: After training is complete, this block prepares for prediction on the test set.
•	It loads the weights of the best performing model (saved in best_path) back into the model.
•	It sets the model to eval() mode.
•	It loops through the test_dl (which yields images and filenames).
•	For each batch, it creates an empty tensor tta_logits to store the sum of predictions.
23.	Inference (TTA) Loop: This is the Test-Time Augmentation (TTA) loop. Instead of predicting on each test image just once, it runs TTA_COUNT (12) times:
•	The first time (i=0), it predicts on the clean, original test image.
•	For the next 11 times, it applies light random augmentations (a bit of noise, slight scaling) to the image and gets another prediction.
•	It adds all these predictions (as probabilities, thanks to torch.softmax) together in tta_logits.
24.	Inference Post-processing: Once the 12 TTA predictions are done for a batch:
•	The summed logits are averaged (/= TTA_COUNT).
•	The final prediction is taken by finding the argmax (the index with the highest average probability).
•	It loops through the batch, matches the name (filename) with the pred (predicted index), converts the index back to a class name (like 'siren') using I2C, and stores the (name, class_name) pair.
25.	Submission File Creation:
•	The all_preds list is converted into a Pandas DataFrame.
•	The DataFrame is saved as submission_optimized.csv in the correct format.
•	It prints a confirmation and a "Distribution" count, which shows you how many test samples were assigned to each class (useful checking).
26.	Download Submission: This final block uses a Google Colab helper function to trigger a download of the submission_optimized.csv file from the Colab environment directly to my local computer's "Downloads" folder.
So this was the things I have used in my code, now below is the explaination of the code of task 2.



